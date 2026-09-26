"""
app.py

TARA AI — Real Production Live Deployment Server.
Exposes real TARA Brain and original trained model inference.
Zero mocks. Zero placeholders. Zero simulated inference.
"""

import os
import sys
import time
import json
import hashlib
from typing import Dict, Any, Optional

# Ensure repository roots are in sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

PROMOTED_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
MODEL_PATH = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
FRONTEND_HTML_PATH = os.path.join(REPO_ROOT, "frontend", "index.html")

print("=" * 70)
print("  TARA AI PRODUCTION SERVER INITIALIZATION")
print("=" * 70)

# 1. Verify Model Integrity
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"CRITICAL: Model file not found at {MODEL_PATH}")

h = hashlib.sha256()
with open(MODEL_PATH, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
computed_hash = h.hexdigest()

print(f"[Model Checksum Audit] Computed: {computed_hash}")
print(f"[Model Checksum Audit] Expected: {PROMOTED_MODEL_SHA256}")
if computed_hash.lower() != PROMOTED_MODEL_SHA256.lower():
    raise ValueError(f"CRITICAL: Model SHA256 mismatch! {computed_hash} != {PROMOTED_MODEL_SHA256}")
print("[OK] Original trained TARA model verified 100% intact.")

# 2. Initialize Real TARA Brain & Runtime
from tara_core.brain import TaraBrain
from tara_model.generate import generate_response

print("Initializing TARA Brain kernel & loading neural model into memory...")
t0 = time.time()
brain = TaraBrain()
load_time = time.time() - t0

if brain.model is None or brain.tokenizer is None:
    raise RuntimeError("CRITICAL: Failed to load trained neural model into TARA Brain.")
print(f"[OK] TARA Brain online with verified model. (Loaded in {load_time:.2f}s)")

# 3. Create FastAPI Production Service
app = FastAPI(
    title="TARA AI Core",
    description="Real Production API for TARA Autonomous Cognitive System",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Reports healthy ONLY when the real TARA model and runtime are verified and live."""
    if brain is None or brain.model is None or brain.tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "UNHEALTHY",
                "error": "TARA Brain or model not initialized.",
                "live_inference_ready": False
            }
        )
    return {
        "status": "HEALTHY",
        "service": "TARA AI Core Production Service",
        "version": "1.1.0",
        "model_identity": "TARA",
        "model_checksum": PROMOTED_MODEL_SHA256,
        "parameter_count": 118080,
        "live_inference_ready": True,
        "timestamp": time.time()
    }


@app.get("/")
@app.get("/chat")
def serve_frontend():
    """Serves the canonical TARA frontend."""
    if not os.path.exists(FRONTEND_HTML_PATH):
        raise HTTPException(status_code=404, detail="Frontend HTML missing")
    with open(FRONTEND_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/endpoints.txt")
def serve_endpoints_config():
    """Serves the dynamic endpoints.txt configuration as plain text."""
    endpoints_txt_path = os.path.join(REPO_ROOT, "endpoints.txt")
    if not os.path.exists(endpoints_txt_path):
        raise HTTPException(status_code=404, detail="endpoints.txt not found")
    with open(endpoints_txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="text/plain; charset=utf-8")


@app.post("/v1/chat")
@app.post("/api/v1/chat")
async def chat_endpoint(request: Request):
    """
    Real production chat endpoint.
    Processes user input through the real TARA Brain kernel,
    evaluating intents, rules, skills, memory, and executing real neural model inference.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    message = (body.get("message") or body.get("prompt") or body.get("input") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing required field 'message' or 'prompt'")

    actor_id = body.get("actor_id") or "user"
    context = body.get("context") or {}

    t_start = time.time()
    try:
        result = brain.process(actor_id=actor_id, input_text=message, context=context)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": f"TARA Brain execution failure: {str(e)}",
                "runtime_stage": "brain.process"
            }
        )

    t_elapsed = time.time() - t_start
    response_text = result.get("final_response") or ""

    return {
        "status": "SUCCESS",
        "response": response_text,
        "text": response_text,
        "model_identity": "TARA",
        "model_checksum": PROMOTED_MODEL_SHA256,
        "parameter_count": 118080,
        "latency_ms": round(t_elapsed * 1000, 2),
        "intent": result.get("intent", {}).get("intent"),
        "decision": result.get("decision"),
        "tool_or_skill": result.get("tool_or_skill"),
        "context_tags": result.get("context_tags", [])
    }


@app.post("/api/v1/inference")
async def direct_inference_endpoint(request: Request):
    """
    Direct autoregressive neural inference through original model.safetensors.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    prompt = (body.get("prompt") or body.get("input") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    max_new_tokens = int(body.get("max_new_tokens", 25))
    temperature = float(body.get("temperature", 0.7))

    t_start = time.time()
    try:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        gen_res = generate_response(
            brain.model,
            brain.tokenizer,
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": f"Neural inference failure: {str(e)}"
            }
        )

    t_elapsed = time.time() - t_start
    text = gen_res.get("text", "")

    return {
        "status": "SUCCESS",
        "text": text,
        "response": text,
        "model_identity": "TARA",
        "model_checksum": PROMOTED_MODEL_SHA256,
        "parameter_count": 118080,
        "token_count": gen_res.get("token_count", 0),
        "tps": gen_res.get("tps", 0.0),
        "latency_ms": round(t_elapsed * 1000, 2)
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\nStarting TARA Production Live Service on http://0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
