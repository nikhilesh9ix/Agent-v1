"""The Agent — a ReAct loop around the LLM: reason, call a tool, observe, repeat
until it produces a text answer. With optional long-term memory."""

from __future__ import annotations

from . import llm
from .memory.base import Memory
from .tools import ToolRegistry


class Agent:
    def __init__(self, system_prompt: str, registry: ToolRegistry | None = None,
                 model: str | None = None, max_iterations: int = 8, verbose: bool = False,
                 episodic_memory: Memory | None = None, semantic_memory: Memory | None = None):
        self.system_prompt = system_prompt
        self.registry = registry or ToolRegistry()
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.messages: list[dict] = []
        self._current_system_prompt = system_prompt  # base prompt + injected memory

    def run(self, user_input: str) -> str:
        """Handle one request, looping through tool calls until an answer."""
        self._current_system_prompt = self._augment_system_prompt(user_input)
        self.messages.append({"role": "user", "content": user_input})

        answer = None
        for _ in range(self.max_iterations):
            message = self._call_llm()
            if message.tool_calls:
                self._handle_tool_calls(message)
                continue
            answer = self._record_final(message.content)
            break
        if answer is None:
            answer = self._summarize_and_stop()

        self._persist_memory(user_input, answer)
        return answer

    def reset(self) -> None:
        """Forget the current conversation (keeps tools and system prompt)."""
        self.messages = []

    # --- loop steps -------------------------------------------------------

    def _payload(self) -> list[dict]:
        return [{"role": "system", "content": self._current_system_prompt}, *self.messages]

    def _call_llm(self, use_tools: bool = True):
        tools = self.registry.openai_schemas() if use_tools else None
        kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        return llm.chat(self._payload(), model=self.model, **kwargs).choices[0].message

    def _handle_tool_calls(self, message) -> None:
        # Assistant's tool-call turn must be in history before the results
        # (provider-safe serialization; Groq rejects the SDK's extra fields).
        self.messages.append(llm.assistant_message_dict(message))
        for call in message.tool_calls:
            if self.verbose:
                print(f"  [tool] {call.function.name}({call.function.arguments})")
            result = self.registry.dispatch(call.function.name, call.function.arguments)
            if self.verbose:
                print(f"  [ -> ] {result[:200]}")
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    def _record_final(self, content: str) -> str:
        self.messages.append({"role": "assistant", "content": content})
        return content

    def _summarize_and_stop(self) -> str:
        """Hit max_iterations: ask for a best-effort answer with tools off."""
        self.messages.append({"role": "user", "content":
            "You have reached the tool-call limit. Stop calling tools and answer now."})
        answer = self._call_llm(use_tools=False).content or ""
        return self._record_final("[stopped after reaching max tool-call iterations] " + answer)

    # --- long-term memory -------------------------------------------------

    def _augment_system_prompt(self, user_input: str) -> str:
        """Prepend relevant recalled memories to the base system prompt."""
        blocks = []
        if self.episodic_memory:
            recent = self.episodic_memory.retrieve(user_input, k=5)
            if recent:
                blocks.append("Recent conversation summaries:\n" + "\n".join(recent))
        if self.semantic_memory:
            facts = self.semantic_memory.retrieve(user_input, k=3)
            if facts:
                blocks.append("Relevant facts:\n" + "\n".join(f"- {f}" for f in facts))
        if not blocks:
            return self.system_prompt
        return self.system_prompt + "\n\n## Memory (background context)\n" + "\n\n".join(blocks)

    def _persist_memory(self, user_input: str, answer: str) -> None:
        """After a run, store an episodic summary and any durable facts."""
        if self.episodic_memory:
            summary = self._summarize_run(user_input, answer)
            self.episodic_memory.add(summary, {"user_query": user_input})
        if self.semantic_memory:
            for fact in self._extract_facts(user_input, answer):
                self.semantic_memory.add(fact, {"source": "conversation"})

    def _summarize_run(self, user_input: str, answer: str) -> str:
        resp = llm.chat([
            {"role": "system", "content": "Summarize the exchange in 2-3 sentences, factual."},
            {"role": "user", "content": f"User asked: {user_input}\n\nAgent answered: {answer}"},
        ], model=self.model)
        return resp.choices[0].message.content.strip()

    def _extract_facts(self, user_input: str, answer: str) -> list[str]:
        """Pull durable facts worth remembering (preferences, stable knowledge)."""
        resp = llm.chat([
            {"role": "system", "content":
                "Extract durable facts worth remembering long-term (preferences, stable "
                "knowledge). Ignore ephemeral details. One fact per line, or exactly NONE."},
            {"role": "user", "content": f"User: {user_input}\n\nAgent: {answer}"},
        ], model=self.model)
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.upper() == "NONE":
            return []
        return [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
