"""
scripts/prepare_and_deploy_hf.py

Prepares and deploys the real production TARA backend and trained model to Hugging Face Spaces:
- Space: manjukp6/tara
- SDK: docker
- Zero mocks, real model.safetensors, real TARA Brain
"""
import os
import shutil
import hashlib
import time
import re
from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(REPO_ROOT, "scratch", "hf_space_bundle")
HF_TOKEN = "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"
REPO_ID = "manjukp6/tara"
PROMOTED_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"

def verify_file_sha256(filepath, expected):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    res = h.hexdigest()
    if res.lower() != expected.lower():
        raise ValueError(f"Checksum mismatch on {filepath}: got {res}, expected {expected}")
    return res

def scan_for_forbidden_terms(directory):
    pattern = re.compile(r'sovereign', re.IGNORECASE)
    violations = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            p = os.path.join(root, f)
            try:
                content = open(p, "r", encoding="utf-8", errors="ignore").read()
                if pattern.search(content):
                    violations.append(p)
            except Exception:
                pass
    if violations:
        raise ValueError(f"CRITICAL: Forbidden terminology found in bundle: {violations}")
    print("[OK] Forbidden terminology scan clean (0 violations).")

def prepare_bundle():
    print(f"Preparing bundle at {BUNDLE_DIR}...")
    if os.path.exists(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    # 1. README.md with docker sdk metadata
    readme_content = """---
title: TARA AI Core
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: TARA Real Production Cognitive System
---

# TARA AI Core — Production Live Deployment

Real production deployment of the TARA cognitive intelligence system.
- Zero mock inference
- Original trained neural model weights (118,080 parameters)
- Model Checksum (SHA-256): `7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309`
- Real-time cognitive evaluation and skill dispatching
"""
    with open(os.path.join(BUNDLE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 2. requirements.txt
    reqs_content = """fastapi>=0.110.0
uvicorn[standard]>=0.28.0
safetensors>=0.4.0
cryptography>=42.0.0
pydantic>=2.0.0
requests>=2.31.0
"""
    with open(os.path.join(BUNDLE_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(reqs_content)

    # 3. Dockerfile
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY frontend/ frontend/
COPY python/ python/
COPY TARA/ TARA/
COPY storage/ storage/

RUN chown -R user:user /app
USER user

ENV PYTHONPATH=/app/python
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""
    with open(os.path.join(BUNDLE_DIR, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    # 4. Copy app.py
    shutil.copy2(os.path.join(REPO_ROOT, "app.py"), os.path.join(BUNDLE_DIR, "app.py"))

    # 5. Copy frontend/
    shutil.copytree(os.path.join(REPO_ROOT, "frontend"), os.path.join(BUNDLE_DIR, "frontend"))

    # 6. Copy python/
    shutil.copytree(os.path.join(REPO_ROOT, "python"), os.path.join(BUNDLE_DIR, "python"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 7. Copy TARA/
    shutil.copytree(os.path.join(REPO_ROOT, "TARA"), os.path.join(BUNDLE_DIR, "TARA"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"))

    # 8. Copy storage/models/tara/
    storage_dest = os.path.join(BUNDLE_DIR, "storage", "models", "tara")
    os.makedirs(storage_dest, exist_ok=True)
    storage_src = os.path.join(REPO_ROOT, "storage", "models", "tara")
    for item in os.listdir(storage_src):
        src_p = os.path.join(storage_src, item)
        dst_p = os.path.join(storage_dest, item)
        if os.path.isfile(src_p):
            shutil.copy2(src_p, dst_p)

    # 9. Verify model integrity in bundle
    bundle_model_path = os.path.join(storage_dest, "model.safetensors")
    verify_file_sha256(bundle_model_path, PROMOTED_MODEL_SHA256)
    print(f"[OK] Bundle model verified SHA-256: {PROMOTED_MODEL_SHA256}")

    # 10. Scan bundle for forbidden terminology
    scan_for_forbidden_terms(BUNDLE_DIR)

    # Compute total bundle size
    total_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(BUNDLE_DIR) for f in fn)
    print(f"[OK] Bundle successfully prepared. Total size: {total_size / (1024*1024):.2f} MB")

def deploy_to_hf():
    api = HfApi(token=HF_TOKEN)
    print(f"Uploading deployment bundle to Hugging Face Space: {REPO_ID}...")
    api.upload_folder(
        folder_path=BUNDLE_DIR,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy real production TARA runtime with neural model.safetensors"
    )
    print("[OK] Upload complete!")

if __name__ == "__main__":
    prepare_bundle()
    deploy_to_hf()
