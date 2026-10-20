from ovos_openai_plugin.api import OpenAIChatCompletions
from ovos_openai_plugin.chat import OpenAIChatEngine
from ovos_openai_plugin.summarizer import OpenAISummarizer
from ovos_openai_plugin.translate import OpenAITextTranslator, OpenAITextLangDetector
from ovos_openai_plugin.dialog_transformers import OpenAIDialogTransformer

# ready-made persona exposed via the "opm.plugin.persona" entry point.
# points the OpenAI chat engine at the public smartgic.io LLama demo server.
LLAMA_DEMO = {
    "name": "Remote LLama",
    "solvers": [
        "ovos-chat-openai-plugin"
    ],
    "ovos-chat-openai-plugin": {
        "api_url": "https://llama.smartgic.io/v1",
        "key": "sk-xxxx",
        "model": "llama3.1:8b"
    }
}

__all__ = [
    "OpenAIChatCompletions",
    "OpenAIChatEngine",
    "OpenAISummarizer",
    "OpenAITextTranslator",
    "OpenAITextLangDetector",
    "OpenAIDialogTransformer",
    "LLAMA_DEMO",
]
