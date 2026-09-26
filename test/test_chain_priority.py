"""The chat engine must carry a priority, or a persona chain cannot be read.

``QuestionSolversService.modules`` sorts the loaded plugins by
``k.priority`` when the caller gives no ``sort_order``, and ``ChatEngine``
declares no default. Without the class attribute the property raised
``AttributeError: 'OpenAIChatEngine' object has no attribute 'priority'``, so
every chain that held this plugin failed, whatever else was configured.
"""
import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole

from ovos_openai_plugin.chat import OpenAIChatEngine

PLUGIN_NAME = "ovos-chat-openai-plugin"


def test_the_engine_declares_a_priority():
    assert isinstance(OpenAIChatEngine.priority, int)
    assert OpenAIChatEngine.priority == 50


def test_it_sorts_before_a_fallback_and_after_an_earlier_engine():
    """50 is the neutral value: ahead of a 9999 fallback, behind a 10 engine."""

    class _Engine(ChatEngine):
        def __init__(self, priority, config=None):
            super().__init__(config or {})
            self.priority = priority

        def continue_chat(self, messages, session_id="default", lang=None,
                          units=None, tools=None):
            return AgentMessage(role=MessageRole.ASSISTANT, content="x")

    engine = OpenAIChatEngine({"api_url": "http://x/v1"})
    earlier, fallback = _Engine(10), _Engine(9999)
    ordered = sorted([fallback, engine, earlier], key=lambda k: k.priority)
    assert ordered == [earlier, engine, fallback]


def test_the_service_loads_it_without_a_sort_order():
    """A chain built the way ovos-persona builds it must be readable."""
    solvers = pytest.importorskip("ovos_persona.solvers")
    # Every installed handler plugin loads, whatever the config names, so
    # another plugin with the same defect would fail this test for the wrong
    # reason. Only this plugin stays enabled.
    others = {name: {"enabled": False}
              for name in solvers.get_utterance_handler_plugins()
              if name != PLUGIN_NAME}
    service = solvers.QuestionSolversService(
        config={PLUGIN_NAME: {"api_url": "http://x/v1"}, **others})
    engines = service.modules
    assert any(isinstance(m, OpenAIChatEngine) for m in engines), engines
