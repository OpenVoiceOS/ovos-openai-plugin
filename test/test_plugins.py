"""Unit tests for summarizer, translator, lang detector and dialog transformer."""
from unittest.mock import patch

from ovos_openai_plugin.summarizer import OpenAISummarizer
from ovos_openai_plugin.translate import OpenAITextTranslator, OpenAITextLangDetector
from ovos_openai_plugin.dialog_transformers import OpenAIDialogTransformer


class TestSummarizer:
    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="short summary")
    def test_summarize(self, mock_req):
        s = OpenAISummarizer({"api_url": "http://x/v1"})
        out = s.summarize("a very long document " * 50)
        assert out == "short summary"
        # the document must be embedded in the user prompt
        sent_messages = mock_req.call_args.args[0]
        assert any("a very long document" in m.content for m in sent_messages)

    def test_custom_template(self):
        s = OpenAISummarizer({"prompt_template": "TLDR: {content}"})
        assert s.prompt_template == "TLDR: {content}"


class TestTranslator:
    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="hola mundo")
    def test_translate_with_explicit_languages(self, mock_req):
        tx = OpenAITextTranslator({"api_url": "http://x/v1"})
        out = tx.translate("hello world", target="es-es", source="en-us")
        assert out == "hola mundo"
        user_msg = mock_req.call_args.args[0][-1].content
        assert "English" in user_msg and "Spanish" in user_msg
        assert "hello world" in user_msg

    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="bonjour")
    def test_translate_target_only(self, mock_req):
        tx = OpenAITextTranslator({"api_url": "http://x/v1"})
        out = tx.translate("hello", target="fr-fr")
        assert out == "bonjour"
        user_msg = mock_req.call_args.args[0][-1].content
        assert "French" in user_msg


class TestLangDetector:
    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="en")
    def test_detect(self, _mock):
        d = OpenAITextLangDetector({"api_url": "http://x/v1"})
        assert d.detect("this is english").startswith("en")

    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="pt")
    def test_detect_probs(self, _mock):
        d = OpenAITextLangDetector({"api_url": "http://x/v1"})
        probs = d.detect_probs("isto é português")
        assert sum(probs.values()) == 1.0
        assert list(probs.values())[0] == 1.0


class TestDialogTransformer:
    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="ARRR matey!")
    def test_transform_with_prompt(self, _mock):
        dt = OpenAIDialogTransformer(config={"api_url": "http://x/v1"})
        out, ctx = dt.transform("hello there", {"prompt": "speak like a pirate"})
        assert out == "ARRR matey!"
        assert ctx == {"prompt": "speak like a pirate"}

    def test_transform_without_prompt_is_noop(self):
        dt = OpenAIDialogTransformer(config={"api_url": "http://x/v1"})
        out, ctx = dt.transform("unchanged dialog")
        assert out == "unchanged dialog"

    @patch("ovos_openai_plugin.api.OpenAIChatCompletions.request", return_value="rewritten")
    def test_rewrite_prompt_from_config(self, _mock):
        dt = OpenAIDialogTransformer(config={"api_url": "http://x/v1",
                                             "rewrite_prompt": "make it formal"})
        out, _ctx = dt.transform("yo")
        assert out == "rewritten"
