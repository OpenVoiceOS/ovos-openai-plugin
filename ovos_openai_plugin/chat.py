from typing import Dict, Optional, List, Iterable, Any

from ovos_plugin_manager.templates.agents import ChatEngine, AgentMessage, MessageRole
from ovos_utils.log import LOG
from sentence_stream import SentenceBoundaryDetector

from ovos_openai_plugin.api import OpenAIChatCompletions


class OpenAIChatEngine(ChatEngine):
    """
    A ChatEngine for OpenAI-compatible Chat Completion APIs.

    Handles multi-turn conversations, system prompt management, native
    tool/function calling, and streaming responses (both raw tokens and complete
    sentences).

    Configuration Dictionary (``config``):
        api_url (str): The endpoint URL (default: "https://api.openai.com/v1").
        key (str): The API key for authentication (optional for local servers).
        model (str): The specific model identifier (e.g., "gpt-4o-mini").
        system_prompt (str): A default system instruction to prepend to chats.
        allow_system_prompts (bool): If True, allows user-provided system messages
                                     to remain or be merged. If False, strips them.

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Knock knock."},
            {"role": "assistant", "content": "Who's there?"},
            {"role": "user", "content": "Orange."},
        ]
    """

    # OpenAI-compatible servers support native function-calling.
    supports_tools = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI Chat Engine.

        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters for the API
                connection and agent behavior.
        """
        super().__init__(config)
        self.api = OpenAIChatCompletions(
            api_url=self.config.get('api_url', 'https://api.openai.com/v1'),
            api_key=self.config.get("key"),
            model=self.config.get("model"),
            config=self.config,
        )
        self.system_prompt = self.config.get("system_prompt")
        self.allow_system = self.config.get("allow_system_prompts") or False
        self._warned_dropped_system = False

    def validate_messages(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        """
        Prepare the message list by enforcing system prompt rules.

        This method:
        1. Strips existing system messages if ``allow_system`` is False.
        2. Injects the configured ``system_prompt`` if it exists.
        3. Merges the configured system prompt with an existing one if
           ``allow_system`` is True.

        Args:
            messages (List[AgentMessage]): The raw input history of messages.

        Returns:
            List[AgentMessage]: The processed list of messages ready for the API.
        """
        if not self.allow_system:
            dropped = [m for m in messages if m.role == MessageRole.SYSTEM]
            if dropped and not self._warned_dropped_system:
                # Context a memory plugin recalled arrives as a system message, and
                # dropping it silently looks like a persona that simply did not
                # remember. Said once per engine, not once per request.
                self._warned_dropped_system = True
                LOG.warning(
                    f"dropped {len(dropped)} system message(s) before the request: "
                    f"'allow_system_prompts' is off. Context a memory plugin injects as a "
                    f"system message is discarded with it; set 'allow_system_prompts', or "
                    f"set the memory plugin's 'inject_mode' to one this engine keeps.")
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if not messages:
            if self.system_prompt:
                return [AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)]
            return []

        if self.system_prompt:
            sysm = AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            if messages[0].role == MessageRole.SYSTEM:
                if self.allow_system:  # merge system prompts
                    sysm = AgentMessage(role=MessageRole.SYSTEM,
                                        content=self.system_prompt + "\n" + messages[0].content)
                # replace existing system prompt
                messages[0] = sysm
            else:
                messages.insert(0, sysm)
        return messages

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: Any = None) -> AgentMessage:
        """
        Generate a complete response message based on the provided chat history.

        Args:
            messages (List[AgentMessage]): Full list of messages in the conversation.
            session_id (str): Identifier for the session (default: "default").
            lang (Optional[str]): BCP-47 language code (e.g., "en-us").
            units (Optional[str]): Preferred unit system (e.g., "metric", "imperial").
            tools: ToolBox object(s) and/or OpenAI tool dicts to expose to the
                model. When provided, the returned message may carry ``tool_calls``.

        Returns:
            AgentMessage: The generated response message from the assistant
            (with ``tool_calls`` populated when the model requests tools).
        """
        messages = self.validate_messages(messages)
        return self.api.chat_message(messages, tools=tools)

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        """
        Stream back response tokens immediately as they are generated.

        Note:
            This yields partial text chunks (tokens) directly from the API.
            These chunks are NOT suitable for direct Text-to-Speech (TTS)
            as they may be incomplete words or punctuation.

        Args:
            messages (List[AgentMessage]): Full list of messages.
            session_id (str): Identifier for the session.
            lang (Optional[str]): Language code.
            units (Optional[str]): Unit system.

        Yields:
            str: A stream of tokens/partial text chunks.
        """
        messages = self.validate_messages(messages)
        yield from self.api.streaming_request(messages)

    def stream_sentences(self, messages: List[AgentMessage],
                         session_id: str = "default",
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> Iterable[str]:
        """
        Stream back response sentences as they are completed.

        This method buffers tokens internally using a SentenceBoundaryDetector
        and only yields text when a grammatical sentence is complete. This
        output IS suitable for direct Text-to-Speech processing.

        Args:
            messages (List[AgentMessage]): Full list of messages.
            session_id (str): Identifier for the session.
            lang (Optional[str]): Language code.
            units (Optional[str]): Unit system.

        Yields:
            str: Complete sentences derived from the token stream.
        """
        messages = self.validate_messages(messages)

        boundary_detector = SentenceBoundaryDetector()

        for tok in self.api.streaming_request(messages):
            yield from boundary_detector.add_chunk(tok)

        final_text = boundary_detector.finish()
        if final_text:
            yield final_text
