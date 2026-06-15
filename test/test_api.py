"""Unit tests for the OpenAIChatCompletions HTTP wrapper."""
import json
from unittest.mock import MagicMock, patch

import pytest
from requests import RequestException

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_openai_plugin.api import OpenAIChatCompletions


def _json_response(payload, status=200):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    resp.status_code = status
    return resp


class TestURLAndHeaders:
    def test_completions_path_appended(self):
        api = OpenAIChatCompletions(api_url="https://example.com/v1")
        assert api.url == "https://example.com/v1/chat/completions"

    def test_trailing_slash_normalized(self):
        api = OpenAIChatCompletions(api_url="https://example.com/v1/")
        assert api.url == "https://example.com/v1/chat/completions"

    def test_default_url_and_model(self):
        api = OpenAIChatCompletions()
        assert api.url == "https://api.openai.com/v1/chat/completions"
        assert api.model == "gpt-4o-mini"

    def test_auth_header_present_with_key(self):
        api = OpenAIChatCompletions(api_key="sk-secret")
        assert api._headers()["Authorization"] == "Bearer sk-secret"

    def test_auth_header_absent_without_key(self):
        api = OpenAIChatCompletions(api_key="")
        assert "Authorization" not in api._headers()


class TestNormalizeMessages:
    def test_agentmessage_normalized(self):
        msgs = [AgentMessage(role=MessageRole.USER, content="hi")]
        out = OpenAIChatCompletions.normalize_messages(msgs)
        assert out == [{"role": "user", "content": "hi"}]

    def test_dicts_passed_through(self):
        msgs = [{"role": "system", "content": "be nice"}]
        assert OpenAIChatCompletions.normalize_messages(msgs) == msgs

    def test_mixed(self):
        msgs = [{"role": "system", "content": "s"},
                AgentMessage(role=MessageRole.USER, content="u")]
        out = OpenAIChatCompletions.normalize_messages(msgs)
        assert out == [{"role": "system", "content": "s"},
                       {"role": "user", "content": "u"}]


class TestRequest:
    @patch("ovos_openai_plugin.api.requests.post")
    def test_returns_message_content(self, mock_post):
        mock_post.return_value = _json_response(
            {"choices": [{"message": {"role": "assistant", "content": "hello there"}}]}
        )
        api = OpenAIChatCompletions(api_url="http://x/v1", api_key="k", model="m")
        out = api.request([AgentMessage(role=MessageRole.USER, content="hi")])
        assert out == "hello there"

    @patch("ovos_openai_plugin.api.requests.post")
    def test_payload_shape(self, mock_post):
        mock_post.return_value = _json_response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        api = OpenAIChatCompletions(api_url="http://x/v1", api_key="k", model="m",
                                    config={"max_tokens": 42, "temperature": 0.1})
        api.request([AgentMessage(role=MessageRole.USER, content="hi")])
        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["model"] == "m"
        assert sent["max_tokens"] == 42
        assert sent["temperature"] == 0.1
        assert sent["messages"] == [{"role": "user", "content": "hi"}]

    @patch("ovos_openai_plugin.api.requests.post")
    def test_model_override(self, mock_post):
        mock_post.return_value = _json_response({"choices": [{"message": {"content": "ok"}}]})
        api = OpenAIChatCompletions(api_url="http://x/v1", model="default")
        api.request([AgentMessage(role=MessageRole.USER, content="hi")], model="override")
        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["model"] == "override"

    @patch("ovos_openai_plugin.api.requests.post")
    def test_error_field_raises(self, mock_post):
        mock_post.return_value = _json_response({"error": "bad key"})
        api = OpenAIChatCompletions(api_url="http://x/v1")
        with pytest.raises(RequestException):
            api.request([AgentMessage(role=MessageRole.USER, content="hi")])


class TestStreaming:
    def _stream_response(self, lines):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = [l.encode("utf-8") for l in lines]
        return resp

    @patch("ovos_openai_plugin.api.requests.post")
    def test_basic_stream(self, mock_post):
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]}),
            '',  # keep-alive newline
            'data: ' + json.dumps({"choices": [{"delta": {"content": "world"}, "finish_reason": None}]}),
            'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            'data: [DONE]',
        ]
        mock_post.return_value = self._stream_response(lines)
        api = OpenAIChatCompletions(api_url="http://x/v1")
        out = list(api.streaming_request([AgentMessage(role=MessageRole.USER, content="hi")]))
        assert out == ["Hello ", "world"]
        assert json.loads(mock_post.call_args.kwargs["data"])["stream"] is True

    @patch("ovos_openai_plugin.api.requests.post")
    def test_done_terminates(self, mock_post):
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "a"}}]}),
            'data: [DONE]',
            'data: ' + json.dumps({"choices": [{"delta": {"content": "should-not-appear"}}]}),
        ]
        mock_post.return_value = self._stream_response(lines)
        api = OpenAIChatCompletions(api_url="http://x/v1")
        out = list(api.streaming_request([AgentMessage(role=MessageRole.USER, content="hi")]))
        assert out == ["a"]

    @patch("ovos_openai_plugin.api.requests.post")
    def test_comment_and_bad_chunks_skipped(self, mock_post):
        lines = [
            ': keep-alive comment',
            'data: not-json',
            'data: ' + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
        ]
        mock_post.return_value = self._stream_response(lines)
        api = OpenAIChatCompletions(api_url="http://x/v1")
        out = list(api.streaming_request([AgentMessage(role=MessageRole.USER, content="hi")]))
        assert out == ["ok"]
