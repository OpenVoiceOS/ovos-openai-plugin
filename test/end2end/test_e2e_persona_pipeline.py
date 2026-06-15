"""Full-pipeline end-to-end test for ovos-openai-plugin using ovoscope.

Proves:
  1. An utterance flows through the real OVOS intent pipeline, hits the
     persona pipeline plugin, reaches the ``OpenAIChatCompletionsSolver``,
     which makes a genuine OpenAI-compatible HTTP round-trip and produces a
     ``speak`` message with non-empty text.
  2. Per-session memory records the USER turn keyed by session_id.

Hermetic: a local FastAPI server implements the OpenAI ``/chat/completions``
contract (both plain JSON and SSE streaming) and returns a deterministic
reply.  The solver is pointed at ``http://127.0.0.1:<port>/v1`` with a dummy
key, so there is no real OpenAI API, no network egress, and no key required.
"""
import json
import os
import socket
import tempfile
import threading
import time
import urllib.request

import pytest

import ovoscope
import ovos_persona

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager

from ovoscope import (
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

assert is_pipeline_available(PERSONA_PIPELINE), (
    "ovos-persona-pipeline-plugin must be installed (ships with ovos-persona)"
)

# ---------------------------------------------------------------------------
# Deterministic reply served by the local OpenAI-compatible stub
# ---------------------------------------------------------------------------

_REPLY = "I am SrvBot and I am happy to help."
_CHUNKS = ["I am ", "SrvBot ", "and ", "I am ", "happy ", "to ", "help."]

PERSONA_NAME = "SrvBot"
PLUGIN_ID = "ovos-solver-openai-plugin"


# ---------------------------------------------------------------------------
# Local OpenAI-compatible server (no key, no external network)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": PERSONA_NAME, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            def event_stream():
                for chunk in _CHUNKS:
                    payload = {
                        "choices": [
                            {"delta": {"content": chunk}, "finish_reason": None}
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(), media_type="text/event-stream"
            )

        return JSONResponse(
            {"choices": [{"message": {"role": "assistant", "content": _REPLY}}]}
        )

    return app


@pytest.fixture(scope="module")
def openai_base_url():
    """Start the local OpenAI-compatible server; yield its ``/v1`` base URL."""
    port = _free_port()
    config = uvicorn.Config(
        _build_app(), host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/models", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("local OpenAI stub did not start in time")

    yield base

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Persona wiring
# ---------------------------------------------------------------------------

def _make_personas_dir(base_url: str) -> str:
    """Write a persona JSON pointing the OpenAI solver at the local stub."""
    tmpdir = tempfile.mkdtemp()
    persona = {
        "name": PERSONA_NAME,
        "solvers": [PLUGIN_ID],
        PLUGIN_ID: {
            "api_url": base_url,
            "key": "not-needed",
            "model": PERSONA_NAME,
            "system_prompt": "You are SrvBot.",
            "enable_memory": True,
        },
    }
    with open(os.path.join(tmpdir, f"{PERSONA_NAME}.json"), "w") as fh:
        json.dump(persona, fh)
    return tmpdir


TEST_PIPELINE = [
    "ovos-persona-pipeline-plugin-high",
    "ovos-persona-pipeline-plugin-low",
]


@pytest.fixture(scope="module")
def mc(openai_base_url):
    """Shared MiniCroft with the OpenAI persona pointed at the local stub."""
    personas_path = _make_personas_dir(openai_base_url)
    pipeline_config = {
        "persona": {
            "personas_path": personas_path,
            "default_persona": PERSONA_NAME,
            "short-term-memory": True,
            "handle_fallback": True,
            "ignore_plugin_personas": True,
        }
    }
    croft = get_minicroft(
        skill_ids=[],
        default_pipeline=TEST_PIPELINE,
        pipeline_config=pipeline_config,
        max_wait=180,
    )
    yield croft
    croft.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _drive_utterance(croft, sess: Session, utterance: str, timeout: int = 30):
    cap = CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
    )
    cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
    return cap.finish()


def _get_persona_service(croft):
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# ---------------------------------------------------------------------------
# Test 1: persona speaks through the full pipeline (real HTTP round-trip)
# ---------------------------------------------------------------------------

class TestOpenAIPersonaSpeaksThroughPipeline:
    def test_pipeline_produces_speak(self, mc):
        sess = Session(session_id="openai-e2e-speak-test")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "who are you", timeout=30)

        msg_types = [m.msg_type for m in messages]
        speak_msgs = [m for m in messages if m.msg_type == "speak"]

        assert speak_msgs, (
            f"Expected at least one 'speak' message; got msg_types: {msg_types}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"'speak' message had an empty utterance; data={speak_msgs[0].data}"
        )

    def test_spoken_text_matches_stub_reply(self, mc):
        sess = Session(session_id="openai-e2e-reply-test")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "what are you", timeout=30)

        spoken = " ".join(
            m.data.get("utterance", "")
            for m in messages
            if m.msg_type == "speak"
        )
        assert "SrvBot" in spoken, (
            f"Expected the stub reply in spoken output, got: {spoken!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: per-session memory records the user turn
# ---------------------------------------------------------------------------

class TestOpenAIPerSessionMemory:
    def test_user_turn_recorded_in_memory(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="openai-e2e-mem-user")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None, f"Persona '{PERSONA_NAME}' not loaded"
        assert persona.memory is not None, "Persona must have short-term memory"

        _drive_utterance(mc, sess, "remember this for me", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        contents = [m.content for m in history]
        assert any("remember this for me" in c for c in contents), (
            f"User utterance not found in memory for session {sess.session_id}. "
            f"History: {contents}"
        )
