"""
scripts/build_cloudflare_worker.py
Compiles the clean Cloudflare edge worker with embedded canonical frontend
and real routing to https://manjukp6-tara.hf.space.
"""
import base64
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(repo_root, "frontend", "index.html")
worker_dest = os.path.join(repo_root, "cloudflare", "src", "index.js")
wrangler_dest = os.path.join(repo_root, "cloudflare", "wrangler.toml")

with open(frontend_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

js_template = """/**
 * Cloudflare Worker: TARA Public Edge & Gateway
 * Worker Name: tara
 *
 * Responsibilities:
 * - Public HTTPS entry point for TARA AI Core
 * - Edge routing to TARA live production backend (Hugging Face Space container)
 * - Zero mock responses. Real end-to-end inference routing.
 */

const CANONICAL_MODEL_IDENTITY = "TARA";
const CANONICAL_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309";
const CANONICAL_PARAM_COUNT = 118080;
const DEFAULT_BACKEND_URL = "https://manjukp6-tara.hf.space";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key, X-Tara-Session-Token",
};

export default {
  async fetch(request, env, ctx) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = url.pathname;
    const backendUrl = env.TARA_CONTROL_PLANE_URL || DEFAULT_BACKEND_URL;

    try {
      // 1. Root & Chat UI
      if (path === "/" || path === "/chat") {
        return handleChatUI(env);
      }

      // 2. Health & Status Endpoints
      if (path === "/health" || path === "/api/v1/health") {
        return forwardToBackend(request, backendUrl, "/health");
      }

      if (path === "/api/v1/control_plane/status") {
        return jsonResponse({
          status: "HEALTHY",
          control_plane: "TARA Dynamic Compute Edge Gateway",
          version: "2.1.0",
          backend_url: backendUrl,
          canonical_model_sha256: CANONICAL_MODEL_SHA256,
          canonical_param_count: CANONICAL_PARAM_COUNT,
          deployment_type: "SERVERLESS_EDGE",
          provider: "cloudflare",
          timestamp: Date.now() / 1000
        });
      }

      // 3. Chat & Inference Endpoints
      if (path === "/v1/chat" || path === "/api/v1/chat") {
        if (request.method !== "POST") {
          return errorResponse("Method not allowed", 405);
        }
        return forwardToBackend(request, backendUrl, "/v1/chat");
      }

      if (path === "/api/v1/inference") {
        if (request.method !== "POST") {
          return errorResponse("Method not allowed", 405);
        }
        return forwardToBackend(request, backendUrl, "/api/v1/inference");
      }

      // 4. Fallback proxy for any other /api/v1/ routes
      if (path.startsWith("/api/v1/")) {
        return forwardToBackend(request, backendUrl, path);
      }

      return errorResponse("Not Found", 404);
    } catch (err) {
      return errorResponse(`Edge Gateway Error: ${err.message}`, 502);
    }
  }
};

async function forwardToBackend(request, backendUrl, targetPath) {
  const base = backendUrl.replace(/\\/+$/, "");
  const target = `${base}${targetPath}`;
  try {
    const backendReq = new Request(target, {
      method: request.method,
      headers: {
        "Content-Type": request.headers.get("Content-Type") || "application/json",
        "Accept": "application/json"
      },
      body: request.method !== "GET" && request.method !== "HEAD" ? await request.text() : undefined
    });

    const res = await fetch(backendReq);
    const data = await res.text();

    return new Response(data, {
      status: res.status,
      headers: {
        "Content-Type": res.headers.get("Content-Type") || "application/json",
        ...CORS_HEADERS
      }
    });
  } catch (e) {
    return jsonResponse({
      status: "ERROR",
      error: `Failed to communicate with TARA production backend: ${e.message}`,
      target_url: target
    }, 502);
  }
}

// Embedded canonical clean Chat UI
const CANONICAL_FRONTEND_B64 = "__B64_FRONTEND__";

function getCanonicalFrontendHTML() {
  const binString = atob(CANONICAL_FRONTEND_B64);
  const bytes = Uint8Array.from(binString, (m) => m.codePointAt(0));
  return new TextDecoder().decode(bytes);
}

function handleChatUI(env) {
  const html = getCanonicalFrontendHTML();
  return new Response(html, {
    headers: { "Content-Type": "text/html;charset=UTF-8", ...CORS_HEADERS }
  });
}

function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: { "Content-Type": "application/json;charset=UTF-8", ...CORS_HEADERS }
  });
}

function errorResponse(message, status = 400) {
  return jsonResponse({ error: message, status: "ERROR" }, status);
}
"""

js_code = js_template.replace("__B64_FRONTEND__", b64)

with open(worker_dest, "w", encoding="utf-8") as f:
    f.write(js_code)

wrangler_toml = """name = "tara"
main = "src/index.js"
compatibility_date = "2024-09-23"

[vars]
PUBLIC_TARA_URL = "https://tara.nawaz.workers.dev"
TARA_CONTROL_PLANE_URL = "https://manjukp6-tara.hf.space"
CANONICAL_MODEL_IDENTITY = "TARA"
CANONICAL_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
CANONICAL_PARAM_COUNT = 118080
USER_COMPUTE_COST = 0.0
"""

with open(wrangler_dest, "w", encoding="utf-8") as f:
    f.write(wrangler_toml)

print("Cloudflare Worker and wrangler.toml built successfully!")
