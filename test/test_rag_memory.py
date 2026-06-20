"""Unit tests for PersonaServerRAGMemory (RAG as a persona memory plugin)."""
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_openai_plugin.rag_memory import PersonaServerRAGMemory

_CFG = {"api_url": "http://x/openai/v1", "vector_store_id": "vs_1"}


def _search_response(*chunks):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [
        {"content": c, "file_id": f"f{i}", "score": 0.9} for i, c in enumerate(chunks)
    ]}
    return resp


class TestConfigValidation:
    def test_requires_api_url(self):
        with pytest.raises(ValueError):
            PersonaServerRAGMemory({"vector_store_id": "vs_1"})

    def test_requires_vector_store_id(self):
        with pytest.raises(ValueError):
            PersonaServerRAGMemory({"api_url": "http://x/v1"})

    def test_rejects_bad_inject_mode(self):
        with pytest.raises(ValueError):
            PersonaServerRAGMemory({**_CFG, "inject_mode": "nope"})


class TestInjectModes:
    @patch("ovos_openai_plugin.rag_memory.requests.post")
    def test_system_mode_separate_message(self, mock_post):
        mock_post.return_value = _search_response("cats are fluffy")
        mem = PersonaServerRAGMemory({**_CFG, "system_prompt": "You are Bob."})
        msgs = mem.build_conversation_context("fluffy animal?", "s1")
        # base system + separate context system + user
        assert msgs[0].role == MessageRole.SYSTEM and msgs[0].content == "You are Bob."
        assert msgs[1].role == MessageRole.SYSTEM and "cats are fluffy" in msgs[1].content
        assert msgs[-1].role == MessageRole.USER and msgs[-1].content == "fluffy animal?"

    @patch("ovos_openai_plugin.rag_memory.requests.post")
    def test_tool_mode(self, mock_post):
        mock_post.return_value = _search_response("cats are fluffy")
        mem = PersonaServerRAGMemory({**_CFG, "inject_mode": "tool"})
        msgs = mem.build_conversation_context("fluffy animal?", "s1")

        # last is the user utterance
        assert msgs[-1].role == MessageRole.USER and msgs[-1].content == "fluffy animal?"
        # an assistant tool_call immediately followed by its TOOL result
        asst = [m for m in msgs if m.role == MessageRole.ASSISTANT and m.tool_calls]
        tool = [m for m in msgs if m.role == MessageRole.TOOL]
        assert asst and tool
        call = asst[0].tool_calls[0]
        assert call.name == "search_knowledge_base"
        assert call.arguments == {"query": "fluffy animal?"}
        assert tool[0].tool_call_id == call.id
        assert "cats are fluffy" in tool[0].content
        # ordering: assistant tool_call precedes its tool result
        assert msgs.index(asst[0]) < msgs.index(tool[0])

    @patch("ovos_openai_plugin.rag_memory.requests.post")
    def test_user_mode_folds_into_utterance(self, mock_post):
        mock_post.return_value = _search_response("ctx")
        mem = PersonaServerRAGMemory({**_CFG, "inject_mode": "user"})
        msgs = mem.build_conversation_context("q?", "s1")
        assert msgs[-1].role == MessageRole.USER
        assert "ctx" in msgs[-1].content and "q?" in msgs[-1].content

    @patch("ovos_openai_plugin.rag_memory.requests.post")
    def test_search_failure_falls_back_to_plain_user(self, mock_post):
        mock_post.side_effect = RuntimeError("server down")
        mem = PersonaServerRAGMemory({**_CFG, "inject_mode": "tool"})
        msgs = mem.build_conversation_context("q?", "s1")
        # no context retrieved → no tool exchange, just the user turn
        assert all(m.role != MessageRole.TOOL for m in msgs)
        assert msgs[-1].role == MessageRole.USER and msgs[-1].content == "q?"

    @patch("ovos_openai_plugin.rag_memory.requests.post")
    def test_min_score_filters_hits(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"data": [
            {"content": "keep", "file_id": "f0", "score": 0.9},
            {"content": "drop", "file_id": "f1", "score": 0.1},
        ]}
        mock_post.return_value = resp
        mem = PersonaServerRAGMemory({**_CFG, "retrieval": {"min_score": 0.5}})
        msgs = mem.build_conversation_context("q?", "s1")
        ctx = " ".join(m.content for m in msgs if m.role == MessageRole.SYSTEM)
        assert "keep" in ctx and "drop" not in ctx
