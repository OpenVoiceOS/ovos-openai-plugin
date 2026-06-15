import json
from typing import Any, Dict, Optional, List, Union, Iterable

import requests
from ovos_plugin_manager.templates.agents import AgentMessage
from ovos_utils.log import LOG
from requests import RequestException

# Type alias for cleaner signatures
MessageList = Union[List[AgentMessage], List[Dict[str, str]]]


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
    def normalize_messages(messages: MessageList) -> List[Dict[str, str]]:
        """
        Convert a list of AgentMessage objects or dicts into the standard OpenAI format.

        Args:
            messages (MessageList): A list containing either AgentMessage objects
                                    or dictionaries.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with 'role' and 'content' keys.
        """
        return [
            {"role": m.role.value, "content": m.content} if isinstance(m, AgentMessage) else m
            for m in messages
        ]

    def _get_common_payload(self, messages: MessageList, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Construct the common JSON payload for API requests.

        Args:
            messages (MessageList): The conversation history.
            model (Optional[str]): The model to use, overriding the default.

        Returns:
            Dict[str, Any]: The configuration dictionary for the API body.
        """
        return {
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

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        return headers

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
        payload = self._get_common_payload(messages, model)

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

        return response["choices"][0]["message"]["content"]

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
