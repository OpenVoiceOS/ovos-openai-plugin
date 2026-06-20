import json
from typing import Any, Dict, Optional, List, Union, Iterable

import requests
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall
from ovos_plugin_manager.templates.agent_tools import ToolBox
from ovos_utils.log import LOG
from requests import RequestException

# Type alias for cleaner signatures
MessageList = Union[List[AgentMessage], List[Dict[str, str]]]


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    """Parse OpenAI tool-call ``arguments`` (a JSON string) into a dict."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        LOG.warning(f"could not parse tool_call arguments: {raw!r}")
        return {}


class OpenAIChatCompletions:
    """
    A thin wrapper around the OpenAI Chat Completions API.

    Handles standard and streaming requests to OpenAI-compatible endpoints,
    managing model parameters and message normalization. Any server exposing
    the ``/chat/completions`` contract (OpenAI, ollama, llama.cpp, vLLM,
    LocalAI, ...) can be used by pointing ``api_url`` at its ``/v1`` base.
    """

    def __init__(self, api_url: str = "https://api.openai.com/v1",
                 api_key: str = "",
                 model: str = "gpt-4o-mini",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API wrapper.

        Args:
            api_url (str): The base URL for the API endpoint (the ``/v1`` root).
                ``/chat/completions`` is appended automatically.
            api_key (str): Authentication key for the API (may be empty for
                local servers that do not require one).
            model (str): The default model identifier to use.
            config (Optional[Dict[str, Any]]): Additional configuration overrides.
        """
        self.config = config or {}
        self.key = api_key or ""
        # the public surface is the /v1 base; the completions path is appended here
        self.url = (api_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.model = model or "gpt-4o-mini"

    @staticmethod
    def normalize_messages(messages: MessageList) -> List[Dict[str, Any]]:
        """
        Convert AgentMessage objects (or dicts) into the OpenAI message format.

        Assistant ``tool_calls`` and ``MessageRole.TOOL`` results are serialized to
        their OpenAI wire shapes so multi-turn tool loops round-trip correctly.

        Args:
            messages (MessageList): A list containing either AgentMessage objects
                                    or dictionaries.

        Returns:
            List[Dict[str, Any]]: OpenAI-format message dicts.
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, AgentMessage):
                out.append(m)  # already a dict
                continue
            d: Dict[str, Any] = {"role": m.role.value, "content": m.content or ""}
            if m.role == MessageRole.TOOL:
                if m.tool_call_id:
                    d["tool_call_id"] = m.tool_call_id
                if m.name:
                    d["name"] = m.name
            if m.tool_calls:
                d["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m.tool_calls
                ]
                # OpenAI expects content=null on an assistant turn that only calls tools
                if not m.content:
                    d["content"] = None
            out.append(d)
        return out

    def _get_common_payload(self, messages: MessageList, model: Optional[str] = None,
                            tools: Any = None) -> Dict[str, Any]:
        """
        Construct the common JSON payload for API requests.

        Args:
            messages (MessageList): The conversation history.
            model (Optional[str]): The model to use, overriding the default.
            tools: ToolBox object(s) and/or OpenAI tool dicts to expose to the
                model; coerced via ``ToolBox.normalize_tools``.

        Returns:
            Dict[str, Any]: The configuration dictionary for the API body.
        """
        payload = {
            "model": model or self.model,
            "messages": self.normalize_messages(messages),
            "max_tokens": self.config.get("max_tokens", 300),
            "temperature": self.config.get("temperature", 0.5),
            "top_p": self.config.get("top_p", 0.2),
            "n": 1,
            "frequency_penalty": self.config.get("frequency_penalty", 0),
            "presence_penalty": self.config.get("presence_penalty", 0),
            "stop": self.config.get("stop_token")
        }
        normalized_tools = ToolBox.normalize_tools(tools)
        if normalized_tools:
            payload["tools"] = normalized_tools
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        return headers

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``payload`` and return the parsed JSON response, raising on error."""
        try:
            resp = requests.post(self.url, headers=self._headers(),
                                 data=json.dumps(payload), timeout=(10, 60))
            resp.raise_for_status()
            response = resp.json()
        except json.JSONDecodeError as e:
            raise RequestException("Failed to decode API response.") from e
        except requests.HTTPError as err:
            raise RequestException(f"HTTP error: {err}") from err

        if "error" in response:
            raise RequestException(response["error"])
        return response

    def request(self, messages: MessageList, model: Optional[str] = None) -> str:
        """
        Send a synchronous chat completion request.

        Args:
            messages (MessageList): The conversation history.
            model (Optional[str]): The model identifier (overrides default).

        Returns:
            str: The content of the assistant's reply.

        Raises:
            RequestException: If the request fails or the API returns an error.
        """
        response = self._post_json(self._get_common_payload(messages, model))
        return response["choices"][0]["message"]["content"]

    def chat_message(self, messages: MessageList, model: Optional[str] = None,
                     tools: Any = None) -> AgentMessage:
        """
        Send a chat completion request and return the full assistant message.

        Unlike :meth:`request` (which returns only text), this preserves any
        ``tool_calls`` the model requests, so callers can run a tool loop.

        Args:
            messages (MessageList): The conversation history.
            model (Optional[str]): The model identifier (overrides default).
            tools: ToolBox object(s) and/or OpenAI tool dicts to expose.

        Returns:
            AgentMessage: assistant message; ``tool_calls`` is populated when the
            model requested tools (``content`` may then be empty).
        """
        response = self._post_json(self._get_common_payload(messages, model, tools))
        msg = response["choices"][0]["message"]
        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = [
                ToolCall(id=tc.get("id") or "",
                         name=tc["function"]["name"],
                         arguments=_parse_arguments(tc["function"].get("arguments")))
                for tc in msg["tool_calls"]
            ]
        return AgentMessage(role=MessageRole.ASSISTANT,
                            content=msg.get("content") or "",
                            tool_calls=tool_calls)

    def streaming_request(self, messages: MessageList, model: Optional[str] = None) -> Iterable[str]:
        """
        Stream response content from the API in real-time.

        Args:
            messages (MessageList): The conversation history.
            model (Optional[str]): The model identifier (overrides default).

        Yields:
            str: Chunks of the assistant's reply text.

        Raises:
            RequestException: If the request fails.
        """
        payload = self._get_common_payload(messages, model)
        payload["stream"] = True

        response = requests.post(self.url, headers=self._headers(), stream=True,
                                 data=json.dumps(payload), timeout=(10, 60))
        try:
            response.raise_for_status()
        except requests.HTTPError as err:
            raise RequestException(f"HTTP error: {err}") from err

        for line in response.iter_lines():
            if not line:
                # keep-alive newline between SSE events
                continue

            line_str = line.decode("utf-8")

            # SSE comment lines (": keep-alive") and anything not a data frame
            if not line_str.startswith("data: "):
                continue

            data_str = line_str.split("data: ", 1)[-1]

            # stream termination signal
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                LOG.error(f"Failed to decode stream chunk: {data_str}")
                continue

            if "error" in chunk:
                if isinstance(chunk["error"], dict) and "message" in chunk["error"]:
                    LOG.error("API returned an error: " + chunk["error"]["message"])
                else:
                    LOG.error(f"API returned an error: {chunk['error']}")
                break

            if not chunk.get("choices"):
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            text = delta.get("content")
            if text:
                yield text

            # honor finish_reason *after* draining any final content
            if choice.get("finish_reason"):
                break
