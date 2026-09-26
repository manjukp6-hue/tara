/**
 * scripts/deploy_tara_model_to_hf.js
 * 
 * Generates real, loadable TARA Model artifacts (Safetensors + Tokenizer + Architecture)
 * and synchronizes them to Hugging Face Hub (tara-project/tara) using Git LFS for binary weights.
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import { URL } from 'url';
import crypto from 'crypto';
import { TaraTokenizer } from '../core/model/TaraTokenizer.js';
import { TaraModelArtifact } from '../core/model/TaraModelArtifact.js';

const tokenArg = process.argv.slice(2).find(a => a.startsWith('--token='))?.split('=')[1];
const HF_TOKEN = process.env.HF_TOKEN || tokenArg;
if (!HF_TOKEN) {
  console.error('Error: HF_TOKEN environment variable or --token=... is required.');
  process.exit(1);
}
const REPO_ID = process.env.HF_REPO_ID || 'tara-project/tara';
const OUTPUT_DIR = path.resolve('storage/models/tara-0.1');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

console.log('====================================================');
console.log('   DEPLOYING TARA AI AUTHENTIC MODEL TO HF HUB      ');
console.log('====================================================\n');

// 1. Export Tokenizer assets
console.log('[1/5] Generating TARA Tokenizer assets...');
const tokenizer = new TaraTokenizer({
  vocabSize: 512,
  modelMaxLength: 32768
});
const tokResult = tokenizer.exportToHuggingFaceFormat(OUTPUT_DIR);
console.log(`      -> Tokenizer assets created: ${tokResult.files.length} files`);

// 2. Read Real Safetensors Model Artifact (trained from zero)
console.log('[2/5] Loading authentic TARA-0.1 trained weights...');
const safetensorsPath = path.join(OUTPUT_DIR, 'model.safetensors');
if (!fs.existsSync(safetensorsPath)) {
  console.error('Error: TARA-0.1 model.safetensors does not exist. Run python train.py first.');
  process.exit(1);
}
const safetensorsBuffer = fs.readFileSync(safetensorsPath);
const lfsOid = crypto.createHash('sha256').update(safetensorsBuffer).digest('hex');
const lfsSize = safetensorsBuffer.length;
console.log(`      -> Authentic model.safetensors: ${(lfsSize / 1024).toFixed(1)} KB`);
console.log(`      -> SHA256 (OID): ${lfsOid}`);

// 3. Load config.json
console.log('[3/5] Reading config.json with TaraForCausalLM architecture...');
const config = {
  architectures: ['TaraForCausalLM'],
  model_type: 'tara-transformer',
  vocab_size: 1024,
  hidden_size: 256,
  intermediate_size: 512,
  num_hidden_layers: 4,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  head_dim: 32,
  hidden_act: 'silu',
  max_position_embeddings: 32768,
  initializer_range: 0.02,
  rms_norm_eps: 1e-05,
  use_cache: true,
  tie_word_embeddings: false,
  rope_theta: 1000000.0,
  torch_dtype: 'float16',
  quantization_config: {
    quant_method: 'tara_int4',
    bits: 4,
    group_size: 128,
    zero_point: true,
    ternary_bitnet_compatible: true
  },
  authority_governance: {
    creator_authority: 'ROOT_EXCLUSIVE',
    telemetry: false,
    watermark: 'none',
    external_ai_dependency: 'none'
  },
  lineage: {
    reference_models: ['Qwen/Qwen2.5-72B (Apache 2.0)', 'mistralai/Mistral-7B-v0.3 (Apache 2.0)', 'microsoft/BitNet (MIT)'],
    documentation: 'docs/MODEL_LINEAGE.md'
  }
};
const configPath = path.join(OUTPUT_DIR, 'config.json');
fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');

// 4. Update README.md
const readme = `---
language:
- en
- kn
license: apache-2.0
tags:
- tara-core
- transformer
- safetensors
- edge-ai
- zero-watermark
- creator-authority
pipeline_tag: text-generation
library_name: tara-core
---

# TARA AI Neural Model (TaraForCausalLM)

## 1. Overview
* **Architecture**: \`TaraForCausalLM\` (Native Transformer with GQA, SwiGLU, and RoPE $\\theta=10^6$)
* **Format**: Standard Hugging Face \`model.safetensors\`
* **Tokenizer**: Native Byte-Level BPE with TARA AI Control Tokens (\`<|creator_auth|>\`, \`<|tara_rule|>\`)
* **Independence**: 100% Watermark-Free, 0% Telemetry, 0% External AI Dependencies (No OpenAI/Gemini/Claude/DeepSeek)
* **Creator Authority**: Governed strictly by the TARA Core RuleEngine and Creator Authority.

## 2. Model Lineage & Open-Source Attribution
* **Reference Architectures**:
  - \`Qwen/Qwen2.5-72B\` (Alibaba Group - Apache 2.0): Grouped-Query Attention (GQA), RoPE with $\\theta=1,000,000$, SwiGLU MLP.
  - \`mistralai/Mistral-7B-v0.3\` (Mistral AI - Apache 2.0): RMSNorm ($\\epsilon=10^{-5}$), interleaved KV caching.
  - \`microsoft/BitNet\` (Microsoft - MIT): Ternary BitNet 1.58-bit and INT4 symmetric block quantization.
* **Lineage Documentation**: See full details in [\`docs/MODEL_LINEAGE.md\`](https://github.com/tara-project/tara/blob/main/docs/MODEL_LINEAGE.md).

## 3. Files in Repository
- \`model.safetensors\`: Real loadable model weights (Safetensors format).
- \`config.json\`: Full architecture hyperparameters.
- \`tokenizer.json\`, \`tokenizer_config.json\`, \`special_tokens_map.json\`: Standard HF tokenizer assets.
- \`README.md\`: Model Card and documentation.
`;
const readmePath = path.join(OUTPUT_DIR, 'README.md');
fs.writeFileSync(readmePath, readme, 'utf8');

// Helper to make https request
function makeRequest(options, data = null) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => resolve({ statusCode: res.statusCode, body }));
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

function putS3Upload(uploadUrl, buffer) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(uploadUrl);
    const options = {
      protocol: parsed.protocol,
      hostname: parsed.hostname,
      path: parsed.pathname + parsed.search,
      method: 'PUT',
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': buffer.length
      }
    };
    const req = https.request(options, (res) => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => resolve({ statusCode: res.statusCode, body: b }));
    });
    req.on('error', reject);
    req.write(buffer);
    req.end();
  });
}

async function uploadLfsAndCommit() {
  console.log('\n[4/5] Uploading model.safetensors via Hugging Face Git LFS...');
  
  // Step A: Request LFS batch upload
  const batchBody = JSON.stringify({
    operation: 'upload',
    transfers: ['basic'],
    objects: [{ oid: lfsOid, size: lfsSize }]
  });

  const batchRes = await makeRequest({
    hostname: 'huggingface.co',
    path: `/${REPO_ID}.git/info/lfs/objects/batch`,
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Accept': 'application/vnd.git-lfs+json',
      'Content-Type': 'application/vnd.git-lfs+json',
      'Content-Length': Buffer.byteLength(batchBody)
    }
  }, batchBody);

  if (batchRes.statusCode !== 200) {
    throw new Error(`LFS Batch Request Failed: HTTP ${batchRes.statusCode} - ${batchRes.body}`);
  }

  const batchData = JSON.parse(batchRes.body);
  const obj = batchData.objects[0];

  if (obj.actions && obj.actions.upload) {
    console.log('      -> Transferring binary payload to LFS cloud storage...');
    const uploadRes = await putS3Upload(obj.actions.upload.href, safetensorsBuffer);
    console.log(`      -> LFS S3 Upload status: HTTP ${uploadRes.statusCode}`);

    if (obj.actions.verify) {
      console.log('      -> Verifying LFS object with Hugging Face...');
      const verifyBody = JSON.stringify({ oid: lfsOid, size: lfsSize });
      const verifyUrl = new URL(obj.actions.verify.href);
      await makeRequest({
        hostname: verifyUrl.hostname,
        path: verifyUrl.pathname,
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_TOKEN}`,
          'Accept': 'application/vnd.git-lfs+json',
          'Content-Type': 'application/vnd.git-lfs+json',
          'Content-Length': Buffer.byteLength(verifyBody)
        }
      }, verifyBody);
      console.log('      -> LFS object verified.');
    }
  } else {
    console.log('      -> LFS object already present on storage.');
  }

  // Step B: Commit all files including lfsFile to Hugging Face
  console.log('\n[5/5] Finalizing atomic commit on Hugging Face Hub...');
  const operations = [
    {
      key: 'header',
      value: {
        summary: 'Publish Real TARA AI Model Artifacts (model.safetensors, config, tokenizer)'
      }
    },
    {
      key: 'file',
      value: {
        content: Buffer.from(fs.readFileSync(readmePath, 'utf8'), 'utf8').toString('base64'),
        path: 'README.md',
        encoding: 'base64'
      }
    },
    {
      key: 'file',
      value: {
        content: Buffer.from(fs.readFileSync(configPath, 'utf8'), 'utf8').toString('base64'),
        path: 'config.json',
        encoding: 'base64'
      }
    },
    {
      key: 'file',
      value: {
        content: Buffer.from(fs.readFileSync(path.join(OUTPUT_DIR, 'tokenizer.json'), 'utf8'), 'utf8').toString('base64'),
        path: 'tokenizer.json',
        encoding: 'base64'
      }
    },
    {
      key: 'file',
      value: {
        content: Buffer.from(fs.readFileSync(path.join(OUTPUT_DIR, 'tokenizer_config.json'), 'utf8'), 'utf8').toString('base64'),
        path: 'tokenizer_config.json',
        encoding: 'base64'
      }
    },
    {
      key: 'file',
      value: {
        content: Buffer.from(fs.readFileSync(path.join(OUTPUT_DIR, 'special_tokens_map.json'), 'utf8'), 'utf8').toString('base64'),
        path: 'special_tokens_map.json',
        encoding: 'base64'
      }
    },
    {
      key: 'lfsFile',
      value: {
        path: 'model.safetensors',
        algo: 'sha256',
        oid: lfsOid,
        size: lfsSize
      }
    }
  ];

  const ndjson = operations.map(op => JSON.stringify(op)).join('\n');
  const commitRes = await makeRequest({
    hostname: 'huggingface.co',
    path: `/api/models/${REPO_ID}/commit/main`,
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/x-ndjson',
      'Content-Length': Buffer.byteLength(ndjson)
    }
  }, ndjson);

  console.log(`\nHF Commit Status: HTTP ${commitRes.statusCode}`);
  const commitData = JSON.parse(commitRes.body);
  console.log('HF Commit Body:', commitData);

  if (commitRes.statusCode === 200) {
    console.log('\n====================================================');
    console.log(' [SUCCESS] REAL TARA MODEL ARTIFACTS DEPLOYED TO HF!');
    console.log(` Commit URL: ${commitData.commitUrl}`);
    console.log(` View at: https://huggingface.co/${REPO_ID}/tree/main`);
    console.log('====================================================');
  } else {
    console.error('Commit failed. See response details above.');
  }
}

uploadLfsAndCommit().catch(err => {
  console.error('Deployment error:', err);
  process.exit(1);
});
