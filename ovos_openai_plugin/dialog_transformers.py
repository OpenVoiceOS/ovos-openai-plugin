from typing import Tuple

from ovos_plugin_manager.templates.transformers import DialogTransformer
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_openai_plugin.api import OpenAIChatCompletions


class OpenAIDialogTransformer(DialogTransformer):
    def __init__(self, name="ovos-dialog-transformer-openai-plugin", priority=10, config=None):
        """
        Initialize the OpenAIDialogTransformer with a name, priority, and configuration.

        Creates an OpenAIChatCompletions client using the configured API key and
        API URL, plus a system prompt from the configuration (or a default prompt
        if not specified).
        """
        super().__init__(name, priority, config)
        self.api = OpenAIChatCompletions(
            api_url=self.config.get('api_url', 'https://api.openai.com/v1'),
            api_key=self.config.get("key"),
            model=self.config.get("model"),
            config=self.config,
        )
        self.system_prompt = self.config.get("system_prompt") or \
                             "Your task is to rewrite text as if it was spoken by a different character"

    def transform(self, dialog: str, context: dict = None) -> Tuple[str, dict]:
        """
        Rewrite the dialog string using a character-specific prompt if available.

        If a prompt is provided in the context or configuration, rewrites the
        dialog as if spoken by a different character; otherwise returns the
        original dialog unchanged.

        Args:
            dialog: The dialog string to be transformed.
            context: Optional dictionary containing transformation context, such
                as a ``prompt``.

        Returns:
            A tuple containing the transformed (or original) dialog and the
            unchanged context.
        """
        context = context or {}
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if not prompt:
            return dialog, context
        return self.api.request([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=f"{prompt} : {dialog}")
        ]), context
