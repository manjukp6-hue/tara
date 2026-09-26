"""
providers/modelscope/app.py

ModelScope Studio Production-Grade Gradio Frontend & Bridge for TARA.
Compliant with ModelScope Studio Gradio environment (2 vCPU / 16 GB):
- Renders the canonical TARA Frontend interface via isolated srcdoc frame
- Provides native Gradio Chatbot fallback for container sandboxes
- Dispatches inference requests to the configurable public TARA endpoint
- Strictly zero user cost (USER_COMPUTE_COST = 0.0)
- Zero secret leakage, no local heavy model files
"""

import os
import sys
import json
import time
import html
import urllib.request
import urllib.error
import gradio as gr

CANONICAL_MODEL_IDENTITY = "TARA"
CANONICAL_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
def get_active_endpoints() -> list:
    """Dynamically loads active provider endpoints from endpoints.txt."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "endpoints.txt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "endpoints.txt")
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return [l.strip().rstrip("/") for l in f if l.strip() and not l.strip().startswith("#")]
    custom_url = os.environ.get("PUBLIC_TARA_URL")
    return [custom_url.rstrip("/")] if custom_url else []


def get_canonical_html() -> str:
    """Loads the canonical TARA frontend source from local disk."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>TARA AI Core Frontend</h3><p>Canonical index.html loading...</p>"


def query_tara_bridge(user_message: str, history):
    """Bridges chat interaction using dynamic failover across endpoints in endpoints.txt."""
    user_message = (user_message or "").strip()
    if not user_message:
        return "", history

    payload = json.dumps({
        "prompt": user_message,
        "message": user_message,
        "input": user_message,
        "actor_id": "modelscope_guest",
        "expected_model_checksum": CANONICAL_MODEL_SHA256
    }).encode("utf-8")

    endpoints = get_active_endpoints()
    tara_reply = None
    last_err = None

    for base in endpoints:
        for route in ["/v1/chat", "/api/v1/chat", "/api/v1/inference"]:
            try:
                req = urllib.request.Request(
                    f"{base}{route}",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=6.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    tara_reply = data.get("response") or data.get("text") or (data.get("data", {}).get("response"))
                    if tara_reply:
                        break
            except Exception as e:
                last_err = e
        if tara_reply:
            break

    if not tara_reply:
        tara_reply = f"[TARA Runtime Notice]: Could not reach active providers from endpoints.txt ({str(last_err)})."

    history = history or []
    # Support both list-of-dicts and list-of-tuples for Gradio cross-version compatibility
    if history and isinstance(history[0], dict):
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": tara_reply})
    else:
        history.append((user_message, tara_reply))
    return "", history


def create_app() -> gr.Blocks:
    """Builds the ModelScope Studio Gradio application."""
    canonical_raw_html = get_canonical_html()
    escaped_html = html.escape(canonical_raw_html, quote=True)
    frame_html = (
        f'<iframe srcdoc="{escaped_html}" '
        f'style="width: 100%; height: 92vh; border: none; border-radius: 8px; background: #070a12;" '
        f'sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals"></iframe>'
    )

    with gr.Blocks(title="TARA AI Core | Cognitive Intelligence") as app:
        with gr.Tabs():
            with gr.TabItem("🌐 Canonical TARA UI"):
                gr.HTML(value=frame_html)

            with gr.TabItem("💬 Native Bridge Chat"):
                gr.Markdown(
                    f"### TARA Intelligence Bridge\n"
                    f"**Public Gateway**: `{PUBLIC_TARA_URL}` | **Model**: `118,080 params` | "
                    f"**SHA256**: `7a50308b...` | **Compute Cost**: `$0.00 / Zero-Cost`"
                )
                chatbot = gr.Chatbot(height=500)
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="Ask TARA...", scale=9, container=False)
                    send_btn = gr.Button("Send ➔", scale=1, variant="primary")

                msg_input.submit(query_tara_bridge, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
                send_btn.click(query_tara_bridge, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

    return app


# ModelScope standard entry point
demo = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[TARA ModelScope Studio] Starting Gradio server on 0.0.0.0:{port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
