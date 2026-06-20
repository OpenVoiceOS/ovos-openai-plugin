# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Retrieval-Augmented Generation as a configurable OVOS persona memory plugin.

``PersonaServerRAGMemory`` is an :class:`AgentContextManager`: before each turn it
searches a vector store hosted by an ``ovos-persona-server`` and injects the
retrieved chunks into the conversation context. The persona's normal chat engine
then generates the answer — RAG composes with any chat backend instead of owning
the chat round-trip.

The plugin is **highly configurable**; every strategy below is selectable from the
persona JSON block (keyed by the plugin name, passed through by ovos-persona):

    {
      "name": "kb-assistant",
      "solvers": ["ovos-chat-openai-plugin"],
      "memory_module": "ovos-openai-rag-memory-plugin",
      "ovos-openai-rag-memory-plugin": {
        "api_url": "http://localhost:8337/openai/v1",
        "vector_store_id": "vs_...",

        "retrieval": {
          "max_num_results": 5,
          "min_score": null,            # drop hits below this score (null = keep all)
          "query_mode": "utterance",    # "utterance" | "history"
          "query_history_turns": 3      # turns folded into the query when query_mode="history"
        },

        "context": {
          "header": "Use the following context to answer ...",
          "chunk_prefix": "- ",
          "chunk_separator": "\\n\\n",
          "include_sources": false,      # prefix each chunk with its source id
          "tool_name": "search_knowledge_base"   # name used for inject_mode="tool"
        },

        "inject_mode": "system",        # system | system_prompt | developer | user | tool
        "system_prompt": "You are a helpful assistant.",
        "max_history": 10,
        "key": "optional-bearer-token"
      }
    }

Injection strategies (`inject_mode`):

- ``system`` (default) — keep the persona's ``system_prompt`` as its own message and
  add the retrieved context as a **separate** system message before the user turn.
  Keeps the base system prompt stable/cacheable.
- ``developer`` — same, but the context goes in a ``developer``-role message.
- ``system_prompt`` — fold context into the persona's system prompt (one combined
  system message) via ``system_prompt_template``.
- ``user`` — prepend the context to the final user message via ``user_template``.
- ``tool`` — present the context as a tool-call result: a synthetic assistant
  ``tool_calls`` turn (a ``search_knowledge_base`` call for the query) followed by a
  ``MessageRole.TOOL`` message carrying the chunks, just before the user utterance.
  Requires a brain/contract with tool-call support (ovos-plugin-manager TOOL role).
"""
from typing import Any, Dict, List, Optional, Tuple

import json
import requests
from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
    ToolCall,
)
from ovos_utils.log import LOG

DEFAULT_HEADER = (
    "Use the following retrieved context to answer the user's question. "
    "If the answer is not in the context, say you don't know."
)
# {system} and {context} for inject_mode="system_prompt"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = "{system}\n\n{header}\n\nContext:\n{context}"
# {context} and {utterance} for inject_mode="user"
DEFAULT_USER_TEMPLATE = "{header}\n\nContext:\n{context}\n\nQuestion: {utterance}"

_VALID_MODES = {"system", "developer", "system_prompt", "user", "tool"}


class PersonaServerRAGMemory(AgentContextManager):
    """Persona memory plugin that augments context with RAG hits from a persona-server."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_url: Optional[str] = self.config.get("api_url")
        self.vector_store_id: Optional[str] = self.config.get("vector_store_id")
        self.key: Optional[str] = self.config.get("key")
        self.max_history: int = self.config.get("max_history", 10)

        retrieval = self.config.get("retrieval", {})
        self.max_num_results: int = retrieval.get("max_num_results", 5)
        self.min_score: Optional[float] = retrieval.get("min_score")
        self.query_mode: str = retrieval.get("query_mode", "utterance")
        self.query_history_turns: int = retrieval.get("query_history_turns", 3)

        ctx = self.config.get("context", {})
        self.header: str = ctx.get("header", DEFAULT_HEADER)
        self.chunk_prefix: str = ctx.get("chunk_prefix", "- ")
        self.chunk_separator: str = ctx.get("chunk_separator", "\n\n")
        self.include_sources: bool = ctx.get("include_sources", False)
        self.tool_name: str = ctx.get("tool_name", "search_knowledge_base")

        self.inject_mode: str = self.config.get("inject_mode", "system")
        self.system_prompt_template: str = self.config.get(
            "system_prompt_template", DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self.user_template: str = self.config.get("user_template", DEFAULT_USER_TEMPLATE)

        self.session2history: Dict[str, List[AgentMessage]] = {}
        self._tool_call_counter: Dict[str, int] = {}

        if not self.api_url:
            raise ValueError("PersonaServerRAGMemory requires 'api_url' in config")
        if not self.vector_store_id:
            raise ValueError("PersonaServerRAGMemory requires 'vector_store_id' in config")
        if self.inject_mode not in _VALID_MODES:
            raise ValueError(f"inject_mode must be one of {sorted(_VALID_MODES)}, "
                             f"got {self.inject_mode!r}")

    # ------------------------------------------------------------------ history
    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Return the retained short-term history for ``session_id``."""
        return list(self.session2history.get(session_id, []))

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """Append ``new_messages`` to the session history, truncating to ``max_history``."""
        history = self.session2history.setdefault(session_id, [])
        history.extend(new_messages)
        if self.max_history:
            self.session2history[session_id] = history[-self.max_history:]

    # ---------------------------------------------------------------- retrieval
    def _build_query(self, utterance: str, session_id: str) -> str:
        """Build the search query per ``query_mode`` (optionally folding in history)."""
        if self.query_mode != "history" or not self.query_history_turns:
            return utterance
        prior_user = [m.content for m in self.get_history(session_id)
                      if m.role == MessageRole.USER][-self.query_history_turns:]
        return " ".join([*prior_user, utterance]).strip()

    def _search(self, query: str) -> List[Tuple[str, str, float]]:
        """Search the vector store; return ``(content, source_id, score)`` per hit."""
        url = f"{self.api_url}/vector_stores/{self.vector_store_id}/search"
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        payload = {"query": query, "max_num_results": self.max_num_results}
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
        hits: List[Tuple[str, str, float]] = []
        for item in resp.json().get("data", []):
            content = item.get("content")
            if not content:
                continue
            if self.min_score is not None and item.get("score", 0.0) < self.min_score:
                continue
            hits.append((content, item.get("file_id", ""), item.get("score", 0.0)))
        return hits

    def _format_context(self, hits: List[Tuple[str, str, float]]) -> str:
        """Render retrieved hits into a single context block string."""
        lines = []
        for content, source_id, _score in hits:
            prefix = self.chunk_prefix
            if self.include_sources and source_id:
                prefix = f"{self.chunk_prefix}[{source_id}] "
            lines.append(f"{prefix}{content}")
        return self.chunk_separator.join(lines)

    def _next_tool_call_id(self, session_id: str) -> str:
        """Return a stable, deterministic tool-call id for ``session_id``."""
        n = self._tool_call_counter.get(session_id, 0)
        self._tool_call_counter[session_id] = n + 1
        return f"rag_{session_id}_{n}"

    # ------------------------------------------------------------------ context
    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Assemble the augmented context per the configured strategies."""
        try:
            hits = self._search(self._build_query(utterance, session_id))
        except Exception as e:
            LOG.error(f"RAG search failed ({e}); proceeding without retrieved context")
            hits = []
        context = self._format_context(hits) if hits else ""

        base_system = self.system_prompt
        messages: List[AgentMessage] = []

        if context and self.inject_mode == "system_prompt":
            combined = self.system_prompt_template.format(
                system=base_system, header=self.header, context=context).strip()
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=combined))
        else:
            if base_system:
                messages.append(AgentMessage(role=MessageRole.SYSTEM, content=base_system))
            if context and self.inject_mode in ("system", "developer"):
                role = MessageRole.DEVELOPER if self.inject_mode == "developer" else MessageRole.SYSTEM
                block = f"{self.header}\n\nContext:\n{context}"
                messages.append(AgentMessage(role=role, content=block))

        messages.extend(self.get_history(session_id))

        # inject_mode="tool": a synthetic search tool-call + result, just before the
        # user turn. Assistant-with-tool_calls precedes its TOOL result (provider
        # ordering invariant); the user utterance stays last (context-manager contract).
        if context and self.inject_mode == "tool":
            call_id = self._next_tool_call_id(session_id)
            messages.append(AgentMessage(
                role=MessageRole.ASSISTANT, content="",
                tool_calls=[ToolCall(id=call_id, name=self.tool_name,
                                     arguments={"query": utterance.strip()})]))
            messages.append(AgentMessage(
                role=MessageRole.TOOL, content=context,
                tool_call_id=call_id, name=self.tool_name))

        if context and self.inject_mode == "user":
            content = self.user_template.format(
                header=self.header, context=context, utterance=utterance.strip())
            messages.append(AgentMessage(role=MessageRole.USER, content=content))
        else:
            messages.append(AgentMessage(role=MessageRole.USER, content=utterance.strip()))

        return messages
