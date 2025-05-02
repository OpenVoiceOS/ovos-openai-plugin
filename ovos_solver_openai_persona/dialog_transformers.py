from typing import Tuple

from ovos_plugin_manager.templates.transformers import DialogTransformer

from ovos_solver_openai_persona import OpenAIChatCompletionsSolver


class OpenAIDialogTransformer(DialogTransformer):
    def __init__(self, name="ovos-dialog-transformer-openai-plugin", priority=10, config=None):
        """
        Initializes the OpenAIDialogTransformer with a name, priority, and configuration.
        
        Creates an OpenAIChatCompletionsSolver using the provided API key, API URL, and a system prompt from the configuration or a default prompt if not specified.
        """
        super().__init__(name, priority, config)
        self.solver = OpenAIChatCompletionsSolver({
            "key": self.config.get("key"),
            'api_url': self.config.get('api_url', 'https://api.openai.com/v1'),
            "enable_memory": False,
            "system_prompt": self.config.get("system_prompt") or "Your task is to rewrite text as if it was spoken by a different character"
        })

    def transform(self, dialog: str, context: dict = None) -> Tuple[str, dict]:
        """
        Transforms the dialog string using a character-specific prompt if available.
        
        If a prompt is provided in the context or configuration, rewrites the dialog as if spoken by a different character using the solver; otherwise, returns the original dialog unchanged.
        
        Args:
            dialog: The dialog string to be transformed.
            context: Optional dictionary containing transformation context, such as a prompt or language.
        
        Returns:
            A tuple containing the transformed (or original) dialog and the unchanged context.
        """
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if not prompt:
            return dialog, context
        return self.solver.get_spoken_answer(f"{prompt} : {dialog}", lang=context.get("lang")), context
