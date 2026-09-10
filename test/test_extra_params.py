"""Provider-specific request parameters, supplied by configuration.

An OpenAI-compatible endpoint is not OpenAI. The parameters that matter differ
per provider and per model, and a payload with a fixed set of fields cannot
express them.
"""
from unittest.mock import patch

import pytest

from ovos_openai_plugin.api import OpenAIChatCompletions


def _api(**config):
    return OpenAIChatCompletions(config=config)


MESSAGES = [{"role": "user", "content": "hello"}]


def test_without_configuration_the_payload_is_unchanged():
    payload = _api()._get_common_payload(MESSAGES)
    assert "extra_params" not in payload
    # the documented defaults still travel, including the one that is None
    assert payload["max_tokens"] == 300
    assert "stop" in payload


def test_a_provider_parameter_reaches_the_payload():
    payload = _api(extra_params={"reasoning_effort": "none"})._get_common_payload(MESSAGES)
    assert payload["reasoning_effort"] == "none"


def test_it_can_override_a_default_the_provider_reads_differently():
    payload = _api(max_tokens=300, extra_params={"max_tokens": 42})._get_common_payload(MESSAGES)
    assert payload["max_tokens"] == 42


def test_none_removes_a_field_instead_of_sending_null():
    payload = _api(extra_params={"stop": None})._get_common_payload(MESSAGES)
    assert "stop" not in payload


@pytest.mark.parametrize("bad", [["reasoning_effort"], [], "", 0, False, "none"])
def test_a_wrong_shape_is_reported_rather_than_fatal(bad):
    """A bad config value must not take the assistant down mid-conversation.

    Falsy wrong types are the ones worth being careful about: treating them as
    "unset" would swallow the mistake instead of reporting it.
    """
    with patch("ovos_openai_plugin.api.LOG") as log:
        payload = _api(extra_params=bad)._get_common_payload(MESSAGES)
    assert payload["max_tokens"] == 300
    log.warning.assert_called_once()


def test_only_an_absent_setting_is_silent():
    """None means unset, and unset is not a mistake."""
    with patch("ovos_openai_plugin.api.LOG") as log:
        _api(extra_params=None)._get_common_payload(MESSAGES)
        _api()._get_common_payload(MESSAGES)
    log.warning.assert_not_called()


@pytest.mark.parametrize("empty", [None, {}])
def test_an_absent_or_empty_setting_changes_nothing(empty):
    assert _api(extra_params=empty)._get_common_payload(MESSAGES) == _api()._get_common_payload(MESSAGES)


def test_the_parameters_survive_into_a_streaming_request():
    """The streaming path builds the same payload, so it must carry them too."""
    api = _api(extra_params={"reasoning_effort": "none"})
    seen = {}

    class _Resp:
        def raise_for_status(self): pass
        def iter_lines(self): return iter([b"data: [DONE]"])

    def _post(url, headers=None, stream=None, data=None, timeout=None):
        import json
        seen.update(json.loads(data))
        return _Resp()

    with patch("ovos_openai_plugin.api.requests.post", _post):
        list(api.streaming_request(MESSAGES))
    assert seen["reasoning_effort"] == "none"
    assert seen["stream"] is True
