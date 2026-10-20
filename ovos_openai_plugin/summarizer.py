from typing import Dict, Optional

from ovos_plugin_manager.templates.agents import SummarizerEngine, AgentMessage, MessageRole

from ovos_openai_plugin.api import OpenAIChatCompletions


class OpenAISummarizer(SummarizerEngine):
    TEMPLATE = """Your task is to summarize the text into a suitable format.
Answer in plaintext with no formatting, 2 paragraphs long at most.
Focus on the most important information.
---------------------
{content}
"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config=config)
        self.api = OpenAIChatCompletions(
            api_url=self.config.get('api_url', 'https://api.openai.com/v1'),
            api_key=self.config.get("key"),
            model=self.config.get("model"),
            config=self.config,
        )
        self.prompt_template = self.config.get("prompt_template") or self.TEMPLATE
        self.system_prompt = self.config.get("system_prompt") or \
                             "Your task is to summarize text in a couple paragraphs."

    def summarize(self, document: str, lang: Optional[str] = None) -> str:
        """
        Create a summary of the provided text.

        Args:
            document (str): The full text to be summarized.
            lang (str, optional): The language of the document.

        Returns:
            str: The summarized text.
        """
        prompt = self.prompt_template.format(content=document)
        return self.api.request([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ])
