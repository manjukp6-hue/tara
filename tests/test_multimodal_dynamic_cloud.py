import os
import sys
import unittest
import tempfile
import torch
import json
import gzip
import csv

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from cloud_train import (
    TaraModel,
    TaraForCausalLM,
    PyTorchVisionEncoder,
    PyTorchAudioEncoder,
    CrossModalProjector,
    TaraStreamingJsonlDataset,
    SPECIAL_TOKENS,
    TaraNativeTokenizer
)


class TestMultimodalDynamicCloud(unittest.TestCase):
    def test_special_tokens(self):
        """Verify required multimodal special tokens exist in SPECIAL_TOKENS vocabulary."""
        for token in ["<|vision_start|>", "<|vision_end|>", "<|audio_start|>", "<|audio_end|>"]:
            self.assertIn(token, SPECIAL_TOKENS, f"Missing special token: {token}")

    def test_multimodal_forward_backward(self):
        """Verify forward and backward passes with text, image, and audio inputs."""
        hidden_size = 64
        vocab_size = 256
        config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 512,
            "vision_config": {
                "patch_height": 8,
                "patch_width": 8,
                "channels": 3,
                "hidden_size": hidden_size,
                "projector_type": "mlp"
            },
            "audio_config": {
                "sample_rate": 16000,
                "n_fft": 256,
                "hop_length": 128,
                "n_mels": 16,
                "hidden_size": hidden_size,
                "projector_type": "linear"
            }
        }

        model = TaraForCausalLM(config)
        input_ids = torch.randint(0, vocab_size, (2, 16))
        images = torch.rand(2, 3, 32, 32)
        audio = torch.rand(2, 512)
        labels = torch.randint(0, vocab_size, (2, 16))

        # Text only
        logits, loss, _, _ = model(input_ids=input_ids, labels=labels)
        self.assertEqual(logits.shape, (2, 16, vocab_size))
        self.assertIsNotNone(loss)
        loss.backward()

        # Text + Images
        model.zero_grad()
        logits_mm, loss_mm, _, _ = model(input_ids=input_ids, images=images, labels=labels)
        self.assertIsNotNone(logits_mm)
        self.assertIsNotNone(loss_mm)
        loss_mm.backward()

        # Text + Audio
        model.zero_grad()
        logits_aud, loss_aud, _, _ = model(input_ids=input_ids, audio=audio, labels=labels)
        self.assertIsNotNone(logits_aud)
        self.assertIsNotNone(loss_aud)
        loss_aud.backward()

    def test_dynamic_rope_extension(self):
        """Verify RoPE table expands dynamically when sequence length exceeds max_position_embeddings."""
        config = {
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 32,  # deliberately small
        }
        model = TaraModel(config)
        self.assertEqual(model.rope_cos.shape[0], 32)

        # Pass input exceeding initial max_pos
        long_input = torch.randint(0, 100, (1, 64))
        out = model(input_ids=long_input)
        self.assertEqual(out.shape, (1, 64, 64))
        self.assertGreaterEqual(model.rope_cos.shape[0], 64)

    def test_extract_training_text_domains(self):
        """Verify _extract_training_text handles heterogeneous domain schemas properly."""
        # 1. COCO Captions (sentences_raw list)
        coco_rec = {"sentences_raw": ["a cat sitting on a mat", "another cat"]}
        _, _, raw = TaraStreamingJsonlDataset._extract_training_text(coco_rec)
        self.assertIn("<|vision_start|>", raw)
        self.assertIn("a cat sitting on a mat", raw)

        # 2. FLEURS Audio transcription
        fleurs_rec = {"transcription": "spoken sentence audio record"}
        _, _, raw_fleurs = TaraStreamingJsonlDataset._extract_training_text(fleurs_rec)
        self.assertIn("<|audio_start|>", raw_fleurs)
        self.assertIn("spoken sentence audio record", raw_fleurs)

        # 3. BFCL Tool Use
        bfcl_rec = {
            "question": [[{"content": "Calculate the square root of 144"}]],
            "function": [{"name": "math_sqrt", "parameters": {"x": 144}}],
            "id": "bfcl_001"
        }
        inp, out, _ = TaraStreamingJsonlDataset._extract_training_text(bfcl_rec)
        self.assertIn("<|tara_exec|>", inp)
        self.assertIn("Calculate the square root of 144", inp)
        self.assertEqual(out, "bfcl_001")

        # 4. Lean 4 Formal Math
        lean_rec = {
            "formal_statement": "theorem add_comm (n m : Nat) : n + m = m + n",
            "proof": "omega"
        }
        inp_l, out_l, _ = TaraStreamingJsonlDataset._extract_training_text(lean_rec)
        self.assertIn("theorem add_comm", inp_l)
        self.assertIn("omega", out_l)

        # 5. CommitPackFT
        commit_rec = {
            "message": "fix: null pointer in parser",
            "diff": "@@ -10 +10 @@\n-if (x == null)\n+if (x != null)"
        }
        inp_c, out_c, _ = TaraStreamingJsonlDataset._extract_training_text(commit_rec)
        self.assertIn("fix: null pointer in parser", inp_c)
        self.assertIn("if (x != null)", out_c)

    def test_dynamic_multiformat_streaming(self):
        """Verify TaraStreamingJsonlDataset streams JSONL, GZ, TXT, CSV, and Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_map = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
            for i in range(len(SPECIAL_TOKENS), 500):
                vocab_map[f"word_{i}"] = i
            vocab_map["test"] = 100
            vocab_map["hello"] = 101
            vocab_map["world"] = 102
            vocab_map["image"] = 103
            tokenizer = TaraNativeTokenizer(vocab_map)

            # JSONL
            jsonl_path = os.path.join(tmpdir, "shard1.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "hello world test"}) + "\n")
                f.write(json.dumps({"input": "hello", "output": "world"}) + "\n")

            # GZ
            gz_path = os.path.join(tmpdir, "shard2.jsonl.gz")
            with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
                gf.write(json.dumps({"text": "hello test from gzip"}) + "\n")

            # TXT
            txt_path = os.path.join(tmpdir, "shard3.txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write("hello world from raw text file paragraph.\n\nsecond paragraph hello.\n")

            # CSV
            csv_path = os.path.join(tmpdir, "shard4.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["id", "content"])
                writer.writerow(["1", "hello world from csv record"])

            # Parquet
            parquet_path = os.path.join(tmpdir, "shard5.parquet")
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pydict({
                "caption": ["a photo of a cat", "a photo of a dog"],
                "image_id": [101, 102]
            })
            pq.write_table(table, parquet_path)

            all_shards = [jsonl_path, gz_path, txt_path, csv_path, parquet_path]
            ds = TaraStreamingJsonlDataset(
                file_paths=all_shards,
                tokenizer=tokenizer,
                max_seq_len=64,
                buffer_size=1
            )
            items = list(ds)
            self.assertGreaterEqual(len(items), 5)

    def test_git_lfs_pointer_detection(self):
        """Verify that Git-LFS pointer stubs are skipped cleanly and do not crash the iterator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lfs_path = os.path.join(tmpdir, "stub.parquet")
            with open(lfs_path, "wb") as f:
                f.write(b"version https://git-lfs.github.com/spec/v1\noid sha256:1234567890abcdef\nsize 12345\n")

            vocab_map = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
            vocab_map["test"] = 100
            tokenizer = TaraNativeTokenizer(vocab_map)

            ds = TaraStreamingJsonlDataset(
                file_paths=[lfs_path],
                tokenizer=tokenizer,
                max_seq_len=64,
                buffer_size=1
            )
            items = list(ds)
            self.assertEqual(len(items), 0, "Git-LFS pointer stub should yield zero items")


if __name__ == "__main__":
    unittest.main()
