# ------------------------------------------------------------
# main.py – Minimal A2A agent that talks to Gemini via AI Studio
# ------------------------------------------------------------
import json
import os
from pathlib import Path

import httpx
import uvicorn
from a2a.types import A2ARequest  # <-- keep if you need the type elsewhere
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, PlainTextResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# ------------------------------------------------------------
# 1 Loads configuration (API key, Agent Card, etc.)
# ------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Set GOOGLE_API_KEY in environment or .env file"
    )

# Path to the static Agent Card – served on /agent-card
AGENT_CARD_PATH = Path(__file__).parent / "agent_card.json"
AGENT_CARD = json.loads(AGENT_CARD_PATH.read_text(encoding="utf-8"))

# ------------------------------------------------------------
# 2 Helper: calls Gemini (Google AI Studio)
# ------------------------------------------------------------
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models"
    "/gemini-2.5-flash-lite:generateContent"
)

async def call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the generated text."""
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GEMINI_ENDPOINT,
            params={"key": GOOGLE_API_KEY},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Gemini returns a nested structure; pull out the first text part
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {data}")

# ------------------------------------------------------------
# 3 A2A request handler (POST /a2a)
# ------------------------------------------------------------
async def a2a_handler(request):
    """JSON‑RPC endpoint that receives a message, forwards it to Gemini,
    and returns a JSON‑RPC response containing the answer."""
    try:
        body = await request.json()
        # ----------------------------------------------------------------
        # Extracts the text part from the JSON‑RPC payload
        # ----------------------------------------------------------------
        params = body.get("params", {})
        message = params.get("message", {})
        parts = message.get("parts", [])

        if not parts:
            raise ValueError("Message must contain at least one part")

        first_part = parts[0]
        if isinstance(first_part, dict):
            message_text = first_part.get("text")
        else:   # very unlikely, but keep the fallback
            message_text = getattr(first_part, "text", None)

        if not isinstance(message_text, str):
            raise ValueError("Message part does not contain a text field")
    except Exception as exc:
        return JSONResponse(
            {"error": f"Invalid request payload: {exc}"},
            status_code=400,
        )

    print(f"🔹 Received prompt: {message_text!r}")

    # --------------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------------
    try:
        completion = await call_gemini(message_text)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Gemini call failed: {exc}"},
            status_code=502,
        )

    print(f"✅ Gemini answer: {completion!r}")

    # --------------------------------------------------------------
    # Build the JSON‑RPC response (another message/send response)
    # --------------------------------------------------------------
    a2a_resp = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "messageId": "response-msg-001",
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": completion,
                    }
                ],
            }
        },
    }
    return JSONResponse(a2a_resp)


# ------------------------------------------------------------
# 4 GET‑only viewer page (HTML UI)
# ------------------------------------------------------------
async def view_handler(request):
    """Serve a tiny HTML page that lets you type a prompt and see the answer."""
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Gemini‑Proxy‑Agent Demo</title>
      <style>
        body {font-family: Arial, Helvetica, sans-serif; margin: 2rem; line-height: 1.5;}
        textarea {width: 100%; height: 120px; font-size: 1rem;}
        button {padding: 0.5rem 1rem; font-size: 1rem; margin-top: 0.5rem;}
        #result {margin-top: 1.5rem; white-space: pre-wrap; background:#f8f8f8; padding:1rem; border:1px solid #ddd;}
      </style>
    </head>
    <body>
      <h1>Gemini‑Proxy‑Agent Demo</h1>
      <p>Enter a prompt, click **Send**, and the answer from Gemini will appear below.</p>

      <textarea id="prompt" placeholder="e.g. Write a short haiku about a rainy city."></textarea><br>
      <button id="sendBtn">Send</button>

      <div id="result"></div>

      <script>
        const btn = document.getElementById('sendBtn');
        const txt = document.getElementById('prompt');
        const out = document.getElementById('result');

        btn.addEventListener('click', async () => {
          const prompt = txt.value.trim();
          if (!prompt) { out.textContent = '❗️ Prompt cannot be empty'; return; }

          // Build the JSON‑RPC payload exactly like the PowerShell example
          const payload = {
            jsonrpc: "2.0",
            method: "message/send",
            params: {
              message: {
                kind: "message",
                messageId: "msg-browser-001",
                role: "user",
                parts: [{ kind: "text", text: prompt }]
              }
            }
          };

          out.textContent = '⏳ Waiting for Gemini…';

          try {
            const resp = await fetch('/a2a', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(payload)
            });

            if (!resp.ok) {
              const err = await resp.json();
              out.textContent = `❌ Error ${resp.status}: ${JSON.stringify(err)}`;
              return;
            }

            const data = await resp.json();
            // Dive into the nested JSON‑RPC response
            const answer = data?.params?.message?.parts?.[0]?.text ?? 'No answer found';
            out.textContent = answer;
          } catch (e) {
            out.textContent = '❌ Network or parsing error: ' + e;
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


# ------------------------------------------------------------
# 5 Endpoint to serve the Agent Card (discovery)
# ------------------------------------------------------------
async def card_handler(request):
    return JSONResponse(AGENT_CARD)


# ------------------------------------------------------------
# 6 Assemble the Starlette app
# ------------------------------------------------------------
routes = [
    Route("/a2a", endpoint=a2a_handler, methods=["POST"]),
    Route("/agent-card", endpoint=card_handler, methods=["GET"]),
    Route("/view", endpoint=view_handler, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],          # <- allow any origin (demo)
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=True, routes=routes, middleware=middleware)


# ------------------------------------------------------------
# 7 Run with Uvicorn (or any ASGI server)
# ------------------------------------------------------------
if __name__ == "__main__":
    # The server is deliberately single‑process & single‑thread 
    uvicorn.run(app, host="0.0.0.0", port=8000)
