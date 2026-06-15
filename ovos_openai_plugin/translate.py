from typing import Optional, Dict, Set

from langcodes import Language
from ovos_config import Configuration
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_utils import classproperty
from ovos_utils.lang import standardize_lang_tag

from ovos_openai_plugin.api import OpenAIChatCompletions


class OpenAITextTranslator(LanguageTranslator):
    def __init__(self, config: Optional[Dict[str, str]] = None):
        super().__init__(config)
        self.api = OpenAIChatCompletions(
            api_url=self.config.get('api_url', 'https://api.openai.com/v1'),
            api_key=self.config.get("key"),
            model=self.config.get("model"),
            config=self.config,
        )
        self.system_prompt = self.config.get("system_prompt") or \
                             "You are a professional translator. Your task is to translate text"

    def translate(self, text: str, target: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Translate the given text from the source language to the target language.

        Args:
            text (str): The text to translate.
            target (Optional[str]): The target language code. If None, the configured language is used.
            source (Optional[str]): The source language code. If None, it is auto-detected.

        Returns:
            str: The translated text.
        """
        target = target or Configuration()["lang"]
        tgt = Language.get(target).display_name('en')
        if source:
            src = Language.get(source).display_name('en')
            prompt = f"Translate the following text from {src} into {tgt}.\n{src}: {text}\n{tgt}: "
        else:
            prompt = f"Translate the following text into {tgt}.\nOriginal: {text}\nTranslated to {tgt}: "
        return self.api.request([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ])


class OpenAITextLangDetector(LanguageDetector):
    def __init__(self, config: Optional[Dict[str, str]] = None):
        super().__init__(config)
        self.api = OpenAIChatCompletions(
            api_url=self.config.get('api_url', 'https://api.openai.com/v1'),
            api_key=self.config.get("key"),
            model=self.config.get("model"),
            config=self.config,
        )
        self.system_prompt = self.config.get("system_prompt") or \
                             ("You are a language guru. Your task is to detect text languages and respond "
                              "with BCP-47 lang codes. You MUST answer ONLY with a language code")

    def detect(self, text: str) -> str:
        return standardize_lang_tag(self.api.request([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=f"Detect the language of this text: {text}")
        ]))

    def detect_probs(self, text: str) -> Dict[str, float]:
        lang = self.detect(text)
        if lang:
            return {lang: 1.0}
        return {}

    @classproperty
    def available_languages(cls) -> Set[str]:
        """
        Return languages supported by this detector implementation.

        The OpenAI-backed detector is not constrained to a fixed language set,
        so this returns an empty set (i.e. "unknown / open set").

        Returns:
            Set[str]: A set of language codes supported by this detector.
        """
        return set()


if __name__ == "__main__":
    cfg = {
        "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "remote_filename": "*q8_0.gguf"
    }
    dt = OpenAITextLangDetector(config=cfg)
    print(dt.detect("The easiest way for anyone to contribute is to help with translations!"))
    # en

    tx = OpenAITextTranslator(config=cfg)
    print(tx.translate("The easiest way for anyone to contribute is to help with translations!",
                       target="es-es"))
    # La forma más sencilla de contribuir es ayudando con las traducciones!
