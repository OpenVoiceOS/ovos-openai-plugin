"""Unit tests for OpenAIChatEngine (system-prompt handling, chat, streaming)."""
from unittest.mock import patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_openai_plugin.chat import OpenAIChatEngine


def _u(text):
    return AgentMessage(role=MessageRole.USER, content=text)


def _s(text):
    return AgentMessage(role=MessageRole.SYSTEM, content=text)


class TestValidateMessages:
    def test_strips_system_when_disallowed(self):
        eng = OpenAIChatEngine({"allow_system_prompts": False})
        out = eng.validate_messages([_s("evil"), _u("hi")])
        assert [m.role for m in out] == [MessageRole.USER]

    def test_injects_configured_system_prompt(self):
        eng = OpenAIChatEngine({"system_prompt": "You are Bob."})
        out = eng.validate_messages([_u("hi")])
        assert out[0].role == MessageRole.SYSTEM
        assert out[0].content == "You are Bob."
        assert out[1].content == "hi"

    def test_empty_messages_with_prompt(self):
        eng = OpenAIChatEngine({"system_prompt": "You are Bob."})
        out = eng.validate_messages([])
        assert len(out) == 1 and out[0].role == MessageRole.SYSTEM

    def test_empty_messages_no_prompt(self):
        eng = OpenAIChatEngine({})
        assert eng.validate_messages([]) == []

    def test_replaces_user_system_when_disallowed(self):
        eng = OpenAIChatEngine({"system_prompt": "Bob", "allow_system_prompts": False})
        out = eng.validate_messages([_s("hacker prompt"), _u("hi")])
        systems = [m.content for m in out if m.role == MessageRole.SYSTEM]
        assert systems == ["Bob"]

    def test_merges_system_when_allowed(self):
        eng = OpenAIChatEngine({"system_prompt": "Bob", "allow_system_prompts": True})
        out = eng.validate_messages([_s("extra"), _u("hi")])
        assert out[0].role == MessageRole.SYSTEM
        assert "Bob" in out[0].content and "extra" in out[0].content


class TestChat:
    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="the answer")
    def test_continue_chat_returns_assistant_message(self, _mock):
        eng = OpenAIChatEngine({"api_url": "http://x/v1"})
        msg = eng.continue_chat([_u("question")])
        assert isinstance(msg, AgentMessage)
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "the answer"

    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.streaming_request",
           return_value=iter(["Hello ", "world"]))
    def test_stream_tokens(self, _mock):
        eng = OpenAIChatEngine({"api_url": "http://x/v1"})
        assert list(eng.stream_tokens([_u("hi")])) == ["Hello ", "world"]

    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.streaming_request")
    def test_stream_sentences_buffers_into_sentences(self, mock_stream):
        # tokens split mid-sentence should be regrouped into full sentences
        mock_stream.return_value = iter(["Hello", " there.", " How", " are", " you?"])
        eng = OpenAIChatEngine({"api_url": "http://x/v1"})
        sentences = list(eng.stream_sentences([_u("hi")]))
        joined = " ".join(sentences)
        assert "Hello there." in joined
        assert "How are you?" in joined
        # each yielded item should be a complete sentence (ends with punctuation)
        assert all(s.strip()[-1] in ".?!" for s in sentences)
