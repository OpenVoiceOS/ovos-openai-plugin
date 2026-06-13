# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end test: OpenAI persona solver → live ovos-persona-server.

Spins up a real ovos-persona-server backed by a mocked Persona (returns a
deterministic answer with no external LLM), then points
``OpenAIChatCompletionsSolver`` at the server's ``/openai/v1`` prefix.
The solver makes a genuine HTTP round-trip to 127.0.0.1; no stubs, no
external API, no key required.

Run::

    pytest test/test_e2e_persona_server.py -v
"""
from __future__ import annotations

import socket
import threading
import time
from unittest.mock import MagicMock

import pytest

# Both packages are optional at the package level — skip the whole module if
# either is absent so the base test suite keeps passing without extras.
pytest.importorskip("ovos_persona_server", reason="ovos-persona-server not installed")
pytest.importorskip("openai", reason="openai SDK not installed")

import httpx
import uvicorn
from fastapi import FastAPI

# The deterministic reply the mocked Persona will return.
_REPLY = "I am SrvBot and I cannot answer that."
_CHUNKS = ["I am ", "SrvBot ", "and ", "I cannot ", "answer ", "that."]


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app using only ovos-persona-server's chat router,
    with a mocked Persona that returns deterministic answers.

    This mirrors the pattern used in ovos-persona-server's own e2e suite
    (tests/e2e/test_e2e_openai.py) to avoid persona API version skew and any
    dependency on an external LLM.
    """
    from ovos_persona_server.chat import chat_router
    import ovos_persona_server.persona as persona_mod
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "srvbot"
    persona.chat.return_value = _REPLY
    persona.stream.side_effect = lambda messages, **kwargs: iter(_CHUNKS)

    persona_mod.default_persona = persona

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


@pytest.fixture(scope="module")
def server_base_url():
    """Start the persona-server in a daemon thread; yield its base URL."""
    port = _free_port()
    app = _build_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/openai/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("persona-server did not start in time")

    yield base

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Test 1 — Direct solver round-trip
# ---------------------------------------------------------------------------

class TestDirectRoundTrip:
    """The solver makes a real HTTP POST to the local persona-server and
    receives back a non-empty string.  No stubs touch the network layer."""

    def test_get_spoken_answer_returns_nonempty_string(self, server_base_url):
        """OpenAIChatCompletionsSolver.get_spoken_answer hits the local server."""
        from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver

        solver = OpenAIChatCompletionsSolver({
            "api_url": f"{server_base_url}/openai/v1",
            "key": "not-needed",
            "model": "srvbot",
            "system_prompt": "You are SrvBot.",
        })

        answer = solver.get_spoken_answer("hello", lang="en-US")

        assert isinstance(answer, str), "expected a string answer"
        assert answer.strip(), "answer must not be blank"

    def test_answer_matches_server_reply(self, server_base_url):
        """The answer returned equals the server's deterministic reply."""
        from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver

        solver = OpenAIChatCompletionsSolver({
            "api_url": f"{server_base_url}/openai/v1",
            "key": "not-needed",
            "model": "srvbot",
            "system_prompt": "You are SrvBot.",
        })

        answer = solver.get_spoken_answer("what are you?", lang="en-US")
        assert answer == _REPLY, f"unexpected answer: {answer!r}"

    def test_memory_records_exchange(self, server_base_url):
        """After a successful answer the solver's internal memory records the QA pair."""
        from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver

        solver = OpenAIChatCompletionsSolver({
            "api_url": f"{server_base_url}/openai/v1",
            "key": "not-needed",
            "model": "srvbot",
            "system_prompt": "You are SrvBot.",
            "enable_memory": True,
        })

        solver.get_spoken_answer("remember me?", lang="en-US")
        assert len(solver.qa_pairs) == 1
        q, a = solver.qa_pairs[0]
        assert q == "remember me?"
        assert a == _REPLY

    def test_continue_chat_with_explicit_messages(self, server_base_url):
        """continue_chat() also makes a real HTTP call and returns a string."""
        from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver

        solver = OpenAIChatCompletionsSolver({
            "api_url": f"{server_base_url}/openai/v1",
            "key": "not-needed",
            "model": "srvbot",
            "system_prompt": "You are SrvBot.",
        })

        messages = [
            {"role": "system", "content": "You are SrvBot."},
            {"role": "user", "content": "ping"},
        ]
        answer = solver.continue_chat(messages, lang="en-US")
        assert isinstance(answer, str) and answer.strip()

    def test_stream_utterances_yields_chunks(self, server_base_url):
        """stream_utterances() streams real SSE chunks from the server."""
        from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver

        solver = OpenAIChatCompletionsSolver({
            "api_url": f"{server_base_url}/openai/v1",
            "key": "not-needed",
            "model": "srvbot",
            "system_prompt": "You are SrvBot.",
        })

        chunks = list(solver.stream_utterances("stream test", lang="en-US"))
        # At least one sentence-ending chunk must come back
        assert chunks, "expected at least one streamed chunk"
        full = " ".join(chunks)
        assert full.strip(), "streamed content must not be blank"
