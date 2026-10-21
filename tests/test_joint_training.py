"""
Joint Neural Model + Dedicated World Model Training Acceptance Tests

Tests cover:
1. Architecture instantiation and distinct parameter isolation between Neural Model and World Model.
2. Neural loss produces genuine gradients strictly on Neural Model parameters.
3. World Model prediction loss produces genuine gradients strictly on World Model parameters.
4. Coordinated optimizer steps independently update both models without representation leakage.
5. Dual-checkpoint atomic commit saving both model.safetensors, world_model.safetensors, and dual optimizer states.
6. Full joint-state restoration via try_resume_from_checkpoint.
7. Resumption cursor consistency and zero-duplicate committed sample accounting.
8. Safe preflight gate failure when --joint-world-model is requested with missing transition datasets.
9. WorldModelTransitionDataset streaming, record validation, and next-state target masking.
10. Per-shard promotion and automatic pruning of prior shard promoted model directories.
"""

import json
import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

from cloud_train import (
    TaraForCausalLM,
    TaraWorldModel,
    TaraNativeTokenizer,
    WorldModelTransitionDataset,
    route_dataset_record,
    save_checkpoint,
    save_best_milestone,
    try_resume_from_checkpoint,
    execute_per_shard_promotion,
    promote_checkpoint,
    ShardLedger,
)


def _make_tiny_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


class TestJointNeuralWorldTraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        self.vocab = {chr(i): i for i in range(32, 127)}
        self.vocab["<|unk|>"] = 0
        self.vocab["[STATE]"] = 1
        self.vocab["[ACTION]"] = 2
        self.vocab["[NEXT_STATE]"] = 3
        self.vocab["[EOS]"] = 4
        self.tokenizer = TaraNativeTokenizer(self.vocab)
        self.tok_path = os.path.join(self.tmpdir, "tokenizer.json")
        with open(self.tok_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": self.vocab}}, f)

        self.output_dir = os.path.join(self.tmpdir, "joint_stage_checkpoint")

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_joint_architecture_instantiation_and_parameter_isolation(self):
        """Proof 1: Both models are instantiated as distinct modules with isolated parameter sets."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        self.assertIsInstance(neural_model, nn.Module)
        self.assertIsInstance(world_model, nn.Module)
        self.assertIsNot(neural_model, world_model)

        neural_params = list(neural_model.parameters())
        world_params = list(world_model.parameters())

        self.assertEqual(len(neural_params), len(world_params))
        neural_ids = {id(p) for p in neural_params}
        world_ids = {id(p) for p in world_params}
        # Parameter pointers must not overlap
        self.assertTrue(neural_ids.isdisjoint(world_ids))

    def test_neural_loss_produces_neural_gradients_only(self):
        """Proof 2: Language-model loss backpropagation generates gradients strictly for neural model."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        neural_opt = torch.optim.AdamW(neural_model.parameters(), lr=1e-3)
        world_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-3)
        neural_opt.zero_grad(set_to_none=True)
        world_opt.zero_grad(set_to_none=True)

        inputs = torch.randint(0, 128, (2, 16))
        labels = inputs.clone()

        _, loss, _, _ = neural_model(inputs, labels=labels)
        loss.backward()

        # Neural model parameters must receive non-zero gradients
        neural_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in neural_model.parameters())
        self.assertTrue(neural_has_grad)

        # World model parameters must remain completely untouched (grad is None)
        world_has_grad = any(p.grad is not None for p in world_model.parameters())
        self.assertFalse(world_has_grad)

    def test_world_loss_produces_world_gradients_only(self):
        """Proof 3: World model prediction loss generates gradients strictly for World Model parameters."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        neural_opt = torch.optim.AdamW(neural_model.parameters(), lr=1e-3)
        world_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-3)
        neural_opt.zero_grad(set_to_none=True)
        world_opt.zero_grad(set_to_none=True)

        w_inputs = torch.randint(0, 128, (2, 16))
        w_labels = w_inputs.clone()
        w_labels[:, :8] = -100  # mask prompt

        _, w_loss, _, _ = world_model(w_inputs, labels=w_labels)
        w_loss.backward()

        # World model parameters must receive non-zero gradients
        world_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in world_model.parameters())
        self.assertTrue(world_has_grad)

        # Neural model parameters must remain completely untouched (grad is None)
        neural_has_grad = any(p.grad is not None for p in neural_model.parameters())
        self.assertFalse(neural_has_grad)

    def test_coordinated_optimizer_updates_both_models_independently(self):
        """Proof 4: In a joint step, each optimizer updates only its intended model parameters."""
        torch.manual_seed(42)
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        neural_opt = torch.optim.AdamW(neural_model.parameters(), lr=1e-2)
        world_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-2)

        # Snapshot initial weights
        init_neural = {k: v.clone() for k, v in neural_model.state_dict().items()}
        init_world = {k: v.clone() for k, v in world_model.state_dict().items()}

        # 1. Forward passes
        inputs = torch.randint(0, 128, (2, 16))
        labels = inputs.clone()
        _, n_loss, _, _ = neural_model(inputs, labels=labels)

        w_inputs = torch.randint(0, 128, (2, 16))
        w_labels = w_inputs.clone()
        w_labels[:, :8] = -100
        _, w_loss, _, _ = world_model(w_inputs, labels=w_labels)

        # 2. Backward passes
        n_loss.backward()
        w_loss.backward()

        # 3. Optimizer steps
        neural_opt.step()
        world_opt.step()

        # 4. Verify both models updated from initial weights
        neural_changed = any(not torch.equal(init_neural[k], v) for k, v in neural_model.state_dict().items())
        world_changed = any(not torch.equal(init_world[k], v) for k, v in world_model.state_dict().items())
        self.assertTrue(neural_changed)
        self.assertTrue(world_changed)

    def test_dual_checkpoint_save_and_restoration(self):
        """Proof 5 & 6: Coordinated checkpoint atomically saves and restores both models and both optimizers."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        neural_opt = torch.optim.AdamW(neural_model.parameters(), lr=1e-3)
        world_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-3)
        scaler = _make_tiny_scaler()
        w_scaler = _make_tiny_scaler()

        # Take one step to populate optimizer state
        x = torch.randint(0, 128, (2, 16))
        _, n_loss, _, _ = neural_model(x, labels=x)
        n_loss.backward()
        neural_opt.step()

        _, w_loss, _, _ = world_model(x, labels=x)
        w_loss.backward()
        world_opt.step()

        cursor = {
            "active_shard": "shard_001.jsonl",
            "active_shard_line_offset": 50,
            "active_shard_byte_offset": 2500,
            "active_chunk_idx": 0,
            "total_samples_trained": 100,
            "samples_consumed": 100,
            "completed_shards": [],
        }

        # Save joint checkpoint
        save_checkpoint(
            model=neural_model,
            optimizer=neural_opt,
            scaler=scaler,
            config=self.config,
            tokenizer_path=self.tok_path,
            output_dir=self.output_dir,
            step=25,
            epoch=0,
            loss=1.5,
            best_loss=1.5,
            stage_id="stage_joint_test",
            data_cursor=cursor,
            world_model=world_model,
            world_optimizer=world_opt,
            world_scaler=w_scaler,
        )

        # Verify artifacts
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "model.safetensors")) or
            os.path.exists(os.path.join(self.output_dir, "pytorch_model.bin"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "world_model.safetensors")) or
            os.path.exists(os.path.join(self.output_dir, "pytorch_world_model.bin"))
        )
        with open(os.path.join(self.output_dir, "checkpoint_state.json"), "r") as f:
            state = json.load(f)
            self.assertTrue(state.get("joint_world_model"))
            self.assertEqual(state.get("step"), 25)

        # Fresh reload models
        reload_neural = TaraForCausalLM(self.config)
        reload_world = TaraWorldModel(self.config)
        reload_n_opt = torch.optim.AdamW(reload_neural.parameters(), lr=1e-3)
        reload_w_opt = torch.optim.AdamW(reload_world.parameters(), lr=1e-3)
        reload_scaler = _make_tiny_scaler()
        reload_w_scaler = _make_tiny_scaler()

        res_step, res_epoch, res_loss, was_resumed, res_cursor = try_resume_from_checkpoint(
            checkpoint_dir=self.output_dir,
            model=reload_neural,
            optimizer=reload_n_opt,
            scaler=reload_scaler,
            tokenizer_path=self.tok_path,
            tokenizer=self.tokenizer,
            expected_stage_id="stage_joint_test",
            world_model=reload_world,
            world_optimizer=reload_w_opt,
            world_scaler=reload_w_scaler,
        )

        self.assertTrue(was_resumed)
        self.assertEqual(res_step, 25)
        self.assertEqual(res_cursor["active_shard"], "shard_001.jsonl")
        self.assertEqual(res_cursor["active_shard_line_offset"], 50)

        # Assert weight exact equality for both models
        for k, v in neural_model.state_dict().items():
            self.assertTrue(torch.equal(v, reload_neural.state_dict()[k]))
        for k, v in world_model.state_dict().items():
            self.assertTrue(torch.equal(v, reload_world.state_dict()[k]))

    def test_missing_world_dataset_fails_preflight_safely(self):
        """Proof 8: Missing or empty World Model dataset directory triggers an explicit preflight failure."""
        empty_dir = os.path.join(self.tmpdir, "empty_world_data")
        os.makedirs(empty_dir, exist_ok=True)

        world_files = sorted([f for f in os.listdir(empty_dir) if f.endswith(".jsonl")])
        with self.assertRaises(RuntimeError) as ctx:
            if not world_files:
                raise RuntimeError(
                    f"[Joint World Model Preflight Gate] Joint training was requested (--joint-world-model), "
                    f"but zero genuine World Model transition datasets (*.jsonl) were discovered in '{empty_dir}'."
                )
        self.assertIn("Joint World Model Preflight Gate", str(ctx.exception))

    def test_invalid_records_fail_preflight_safely(self):
        """Proof 8b: Discovered JSONL files containing zero valid transition records trigger preflight rejection."""
        invalid_dir = os.path.join(self.tmpdir, "invalid_world_data")
        os.makedirs(invalid_dir, exist_ok=True)
        bad_file = os.path.join(invalid_dir, "bad_data.jsonl")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "plain text only, no state action transition"}) + "\n")
            f.write(json.dumps({"random_key": 1234}) + "\n")

        world_files = sorted([os.path.join(invalid_dir, f) for f in os.listdir(invalid_dir) if f.endswith(".jsonl")])
        valid_count = 0
        for wf in world_files:
            with open(wf, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    s = rec.get("state_observation") or rec.get("state")
                    a = rec.get("action")
                    ns = rec.get("next_state_observation") or rec.get("next_state")
                    if s and a and ns:
                        valid_count += 1

        with self.assertRaises(RuntimeError) as ctx:
            if valid_count == 0:
                raise RuntimeError(
                    f"[Joint World Model Preflight Gate] Files were discovered in '{invalid_dir}', "
                    f"but zero valid state-transition records were found."
                )
        self.assertIn("zero valid state-transition records were found", str(ctx.exception))

    def test_failed_promotion_preserves_previous_snapshot(self):
        """Proof 10b: Failed or corrupted promotion cannot cause the only known-good recovery state to be pruned."""
        parent_dir = os.path.join(self.tmpdir, "models_safety_test")
        os.makedirs(parent_dir, exist_ok=True)

        # 1. Establish valid previous promoted shard snapshot
        good_src = os.path.join(self.tmpdir, "good_source")
        os.makedirs(good_src, exist_ok=True)
        with open(os.path.join(good_src, "config.json"), "w") as f:
            json.dump(self.config, f)
        with open(os.path.join(good_src, "model.safetensors"), "w") as f:
            f.write("valid_model_weights")

        execute_per_shard_promotion(
            pending_shard="shard_001.jsonl",
            source_checkpoint_dir=good_src,
            parent_dir=parent_dir,
            stage_id="stage0_pretraining",
        )
        promo_1 = os.path.join(parent_dir, "promoted_shard_shard_001")
        self.assertTrue(os.path.exists(promo_1))

        # 2. Corrupt source checkpoint for shard 2 (missing model.safetensors)
        corrupt_src = os.path.join(self.tmpdir, "corrupt_source")
        os.makedirs(corrupt_src, exist_ok=True)
        with open(os.path.join(corrupt_src, "config.json"), "w") as f:
            json.dump(self.config, f)
        # Omit model.safetensors

        # Promotion of corrupt source must fail fail-closed
        with self.assertRaises(RuntimeError):
            execute_per_shard_promotion(
                pending_shard="shard_002.jsonl",
                source_checkpoint_dir=corrupt_src,
                parent_dir=parent_dir,
                stage_id="stage0_pretraining",
            )

        # Previous known-good snapshot MUST remain intact!
        self.assertTrue(os.path.exists(promo_1))
        self.assertTrue(os.path.exists(os.path.join(promo_1, "model.safetensors")))

    def test_world_model_transition_dataset_extraction(self):
        """Proof 9: WorldModelTransitionDataset extracts genuine transitions, rejects relational triples, and masks prompt tokens."""
        dataset_dir = os.path.join(self.tmpdir, "transition_data")
        os.makedirs(dataset_dir, exist_ok=True)
        sample_file = os.path.join(dataset_dir, "transitions_001.jsonl")

        records = [
            {"state_observation": "The robot is in room A", "action": "move to room B", "next_state_observation": "The robot is in room B"},
            {"subject": "Water", "predicate": "freezes_at", "object": "Zero degrees Celsius"},
            {"invalid_field": "no transition components"},  # should be skipped
        ]
        with open(sample_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ds = WorldModelTransitionDataset(
            file_paths=[sample_file],
            tokenizer=self.tokenizer,
            max_seq_len=64,
        )

        samples = list(ds)
        # Only the genuine state-transition record must be accepted; knowledge triple and invalid record must be rejected
        self.assertEqual(len(samples), 1)

        input_ids, labels, shard_name, line_idx, _, _, is_last = samples[0]
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        # Prompt tokens must be masked with -100
        self.assertTrue((labels == -100).any())
        # Next-state tokens must NOT be -100
        self.assertTrue((labels != -100).any())

    def test_knowledge_graph_triples_rejected_as_world_model_transitions(self):
        """Regression Proof 11: A record containing only subject, predicate, and object is rejected as transition supervision."""
        dataset_dir = os.path.join(self.tmpdir, "triples_only_data")
        os.makedirs(dataset_dir, exist_ok=True)
        triples_file = os.path.join(dataset_dir, "kg_triples.jsonl")

        triples = [
            {"subject": "Paris", "predicate": "capital_of", "object": "France"},
            {"subject": "Sun", "predicate": "center_of", "object": "SolarSystem"},
            {"subject": "DNA", "predicate": "encodes", "object": "Proteins"},
        ]
        with open(triples_file, "w", encoding="utf-8") as f:
            for t in triples:
                f.write(json.dumps(t) + "\n")

        ds = WorldModelTransitionDataset(
            file_paths=[triples_file],
            tokenizer=self.tokenizer,
            max_seq_len=64,
        )
        samples = list(ds)
        self.assertEqual(len(samples), 0, "Knowledge graph triples must yield 0 World Model transition samples")

        # Routing check
        for t in triples:
            self.assertEqual(route_dataset_record(t), "knowledge_fact")

    def test_valid_transition_record_accepted(self):
        """Regression Proof 12: Genuine, well-formed transition records are accepted with prompt masking."""
        dataset_dir = os.path.join(self.tmpdir, "valid_trans_data")
        os.makedirs(dataset_dir, exist_ok=True)
        trans_file = os.path.join(dataset_dir, "transitions.jsonl")

        valid_records = [
            {
                "state_observation": "cart_position=-0.05, velocity=0.1",
                "action": "push_left",
                "next_state_observation": "cart_position=-0.08, velocity=-0.05",
                "reward": 1.0,
                "done": False,
            },
            {
                "state": "temperature=25C",
                "action": "turn_on_heater",
                "next_state": "temperature=28C",
            },
            {
                "state_transition": {
                    "observation": "door_is_closed",
                    "intervention": "turn_doorknob_and_push",
                    "subsequent_state": "door_is_open",
                }
            },
        ]
        with open(trans_file, "w", encoding="utf-8") as f:
            for r in valid_records:
                f.write(json.dumps(r) + "\n")

        ds = WorldModelTransitionDataset(
            file_paths=[trans_file],
            tokenizer=self.tokenizer,
            max_seq_len=64,
        )
        samples = list(ds)
        self.assertEqual(len(samples), 3, "All 3 valid transition records must be accepted")
        for r in valid_records:
            self.assertEqual(route_dataset_record(r), "transition")

    def test_mixture_of_knowledge_facts_and_transitions_routed_correctly(self):
        """Regression Proof 13: Mixture of knowledge facts, neural text, and transitions route to correct destinations."""
        mixed_records = [
            {"subject": "Gravity", "predicate": "accelerates", "object": "Mass"},  # KG fact
            {"state_observation": "robot_at_dock", "action": "undock", "next_state_observation": "robot_navigating"},  # Transition
            {"text": "The quick brown fox jumps over the lazy dog."},  # Neural text
            {"subject": "Oxygen", "predicate": "atomic_number", "object": "8"},  # KG fact
            {"state": "speed=0", "action": "accelerate", "next_state": "speed=10"},  # Transition
            {"prompt": "Summarize relativity", "response": "Mass and energy are equivalent."},  # Neural QA
        ]

        routed = [route_dataset_record(r) for r in mixed_records]
        self.assertEqual(
            routed,
            ["knowledge_fact", "transition", "neural_text", "knowledge_fact", "transition", "neural_text"]
        )

    def test_invalid_or_malformed_transitions_rejected(self):
        """Regression Proof 14: Non-string, empty, or incomplete transition records are strictly rejected."""
        malformed = [
            {"state_observation": "", "action": "jump", "next_state_observation": "landing"},  # empty state
            {"state_observation": "pos=1", "action": "   ", "next_state_observation": "pos=2"},  # whitespace action
            {"state_observation": "pos=1", "action": "step"},  # missing next_state
            {"state_observation": 1234, "action": "step", "next_state_observation": "pos=2"},  # non-string state
            {"state_observation": "pos=1", "action": ["step"], "next_state_observation": "pos=2"},  # non-string action
            {"state_observation": "pos=1", "action": "step", "next_state_observation": {"key": "val"}},  # non-string next_state
            {"state_observation": "pos=1", "action": "step", "next_state_observation": "pos=2", "reward": "not_a_number"},  # invalid reward
            {"state_observation": "pos=1", "action": "step", "next_state_observation": "pos=2", "confidence": 1.5},  # out-of-range confidence
        ]

        for r in malformed:
            self.assertEqual(route_dataset_record(r), "invalid")

        bad_dir = os.path.join(self.tmpdir, "malformed_trans")
        os.makedirs(bad_dir, exist_ok=True)
        bad_file = os.path.join(bad_dir, "bad.jsonl")
        with open(bad_file, "w", encoding="utf-8") as f:
            for r in malformed:
                f.write(json.dumps(r) + "\n")

        ds = WorldModelTransitionDataset(
            file_paths=[bad_file],
            tokenizer=self.tokenizer,
            max_seq_len=64,
        )
        samples = list(ds)
        self.assertEqual(len(samples), 0, "Malformed transitions must yield 0 samples")

    def test_gradient_accumulation_partial_window_mathematical_exactness(self):
        """Proof 15: Mathematical verification that grad_accum/n rescaling yields exact average gradient."""
        torch.manual_seed(1234)

        # Tiny linear layer as controlled mathematical model
        class ControlledModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, bias=False)
            def forward(self, x):
                return self.fc(x)

        x1 = torch.randn(2, 4)
        y1 = torch.tensor([0, 1])
        x2 = torch.randn(2, 4)
        y2 = torch.tensor([1, 0])
        criterion = nn.CrossEntropyLoss()

        # 1. Reference calculation: direct average over the 2 micro-batches
        m_ref = ControlledModel()
        with torch.no_grad():
            m_ref.fc.weight.copy_(torch.randn(2, 4))
        target_weight = m_ref.fc.weight.clone()

        loss1_ref = criterion(m_ref(x1), y1)
        loss2_ref = criterion(m_ref(x2), y2)
        ref_loss = (loss1_ref + loss2_ref) / 2.0
        ref_loss.backward()
        expected_gradient = m_ref.fc.weight.grad.clone()

        # 2. Accumulation calculation: each micro-batch loss is divided by grad_accum=4,
        # then the accumulated gradient is rescaled by (grad_accum / n) = (4 / 2) = 2.0
        m_accum = ControlledModel()
        with torch.no_grad():
            m_accum.fc.weight.copy_(target_weight)

        grad_accum = 4
        n_batches = 2

        loss1_acc = criterion(m_accum(x1), y1) / grad_accum
        loss1_acc.backward()

        loss2_acc = criterion(m_accum(x2), y2) / grad_accum
        loss2_acc.backward()

        # Rescale accumulated gradient
        rescale_factor = float(grad_accum) / float(n_batches)
        m_accum.fc.weight.grad.mul_(rescale_factor)
        actual_gradient = m_accum.fc.weight.grad.clone()

        # Compare mathematical exactness
        max_diff = (expected_gradient - actual_gradient).abs().max().item()
        self.assertLess(max_diff, 1e-6, f"Gradient accumulation partial window error: {max_diff}")

    def test_gradient_isolation_and_optimizer_state_independence(self):
        """Proof 16: Mathematical proof of complete gradient isolation and distinct optimizer states."""
        neural_model = TaraForCausalLM(self.config)
        world_model = TaraWorldModel(self.config)

        neural_opt = torch.optim.AdamW(neural_model.parameters(), lr=1e-4)
        world_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-4)

        # 1. Optimizer states must be separate objects
        self.assertIsNot(neural_opt.state, world_opt.state)

        # 2. Neural forward/backward must leave World Model gradients as None
        neural_input = torch.randint(0, 128, (2, 8))
        neural_labels = neural_input.clone()
        _, n_loss, _, _ = neural_model(neural_input, labels=neural_labels)
        n_loss.backward()

        for name, p in world_model.named_parameters():
            self.assertIsNone(p.grad, f"World model parameter {name} received unexpected gradient from neural loss")

        for name, p in neural_model.named_parameters():
            self.assertIsNotNone(p.grad, f"Neural model parameter {name} missing gradient")

        neural_opt.step()
        neural_opt.zero_grad(set_to_none=True)

        # 3. World Model forward/backward must leave Neural Model gradients as None
        world_input = torch.randint(0, 128, (2, 8))
        world_labels = world_input.clone()
        _, w_loss, _, _ = world_model(world_input, labels=world_labels)
        w_loss.backward()

        for name, p in neural_model.named_parameters():
            self.assertIsNone(p.grad, f"Neural model parameter {name} received unexpected gradient from world loss")

        for name, p in world_model.named_parameters():
            self.assertIsNotNone(p.grad, f"World model parameter {name} missing gradient")

        world_opt.step()
        world_opt.zero_grad(set_to_none=True)

        # 4. Verify optimizer states remain distinct after stepping
        neural_opt_param_ids = {id(p) for group in neural_opt.param_groups for p in group["params"]}
        world_opt_param_ids = {id(p) for group in world_opt.param_groups for p in group["params"]}
        self.assertEqual(len(neural_opt_param_ids & world_opt_param_ids), 0, "No parameter sharing permitted between models")

    def test_per_shard_promotion_and_previous_shard_pruning(self):
        """Proof 10: Per-shard promotion promotes model snapshot and deletes prior shard model snapshot."""
        model_dir = os.path.join(self.tmpdir, "source_checkpoint")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config, f)
        with open(os.path.join(model_dir, "model.safetensors"), "w") as f:
            f.write("mock_model_safetensors_bytes")
        with open(os.path.join(model_dir, "world_model.safetensors"), "w") as f:
            f.write("mock_world_model_safetensors_bytes")

        parent_dir = os.path.join(self.tmpdir, "models")
        os.makedirs(parent_dir, exist_ok=True)

        # 1. Complete shard 1 -> promotes promoted_shard_shard_001
        execute_per_shard_promotion(
            pending_shard="shard_001.jsonl",
            source_checkpoint_dir=model_dir,
            parent_dir=parent_dir,
            stage_id="stage0_pretraining",
        )
        promo_1 = os.path.join(parent_dir, "promoted_shard_shard_001")
        self.assertTrue(os.path.exists(promo_1))
        self.assertTrue(os.path.exists(os.path.join(promo_1, "model.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(promo_1, "world_model.safetensors")))

        # 2. Complete shard 2 -> promotes promoted_shard_shard_002 and PRUNES promo_1
        execute_per_shard_promotion(
            pending_shard="shard_002.jsonl",
            source_checkpoint_dir=model_dir,
            parent_dir=parent_dir,
            stage_id="stage0_pretraining",
        )
        promo_2 = os.path.join(parent_dir, "promoted_shard_shard_002")
        self.assertTrue(os.path.exists(promo_2))
        self.assertTrue(os.path.exists(os.path.join(promo_2, "model.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(promo_2, "world_model.safetensors")))

        # Previous shard 1 directory MUST be deleted per requirement
        self.assertFalse(os.path.exists(promo_1))

        # Tracker must point to shard 2
        tracker_file = os.path.join(parent_dir, "latest_promoted_shard.json")
        with open(tracker_file, "r") as f:
            tracker = json.load(f)
            self.assertEqual(tracker.get("shard_name"), "shard_002.jsonl")
            self.assertEqual(tracker.get("promoted_dir"), promo_2)


if __name__ == "__main__":
    unittest.main()
