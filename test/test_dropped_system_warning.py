"""Dropping recalled context must be visible in the log.

A memory plugin returns what it recalled as a system message. This engine
removes system messages unless ``allow_system_prompts`` is set, so on the
default path the recalled context is discarded and the persona answers as
though it never remembered. The engine says so once, naming the setting.

The assertions patch ``LOG`` rather than read a captured log. ``ovos_utils.log``
gives every call site its own logger and turns propagation off, so no pytest
fixture observes one of its lines; the line an operator sees was checked by
hand and is quoted in the audit note beside this repository.
"""
from unittest.mock import patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_openai_plugin.chat import OpenAIChatEngine

_CFG = {"api_url": "http://x/v1", "key": "none", "model": "m"}


def _engine(**cfg):
    return OpenAIChatEngine(config={**_CFG, **cfg})


def _messages():
    return [AgentMessage(MessageRole.SYSTEM, "recalled: the secret word is marmalade"),
            AgentMessage(MessageRole.USER, "what is the secret word")]


def test_warns_when_it_drops_a_system_message():
    with patch("ovos_openai_plugin.chat.LOG") as log:
        kept = _engine().validate_messages(_messages())
    assert [m.role for m in kept] == [MessageRole.USER]
    said = " ".join(str(c) for c in log.warning.call_args_list)
    assert "allow_system_prompts" in said and "inject_mode" in said, said


def test_warns_once_per_engine():
    engine = _engine()
    with patch("ovos_openai_plugin.chat.LOG") as log:
        engine.validate_messages(_messages())
        engine.validate_messages(_messages())
    assert log.warning.call_count == 1


def test_no_warning_when_system_messages_are_kept():
    with patch("ovos_openai_plugin.chat.LOG") as log:
        kept = _engine(allow_system_prompts=True).validate_messages(_messages())
    assert [m.role for m in kept] == [MessageRole.SYSTEM, MessageRole.USER]
    assert log.warning.call_count == 0


def test_no_warning_without_a_system_message():
    with patch("ovos_openai_plugin.chat.LOG") as log:
        _engine().validate_messages([AgentMessage(MessageRole.USER, "hello")])
    assert log.warning.call_count == 0
