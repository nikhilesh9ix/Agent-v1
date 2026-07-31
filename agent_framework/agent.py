"""Phase 4: the Agent — a loop around the LLM.

An agent is not a smarter model; it's a while-loop with an LLM inside, wired to
tools. Each turn the model either (a) asks for a tool call, which we run and feed
back, or (b) produces a final text answer, which ends the loop. `max_iterations`
is the safety belt so a model that keeps calling tools forever can't burn time
and money without bound.

run() is deliberately thin — it just drives the loop. The real steps live in
small, single-purpose methods so each is easy to read, test, and change.
"""

from __future__ import annotations

from . import llm
from .memory.base import Memory
from .tools import ToolRegistry


class Agent:
    """Runs the ReAct loop: reason -> act (tool) -> observe -> repeat -> answer."""

    def __init__(self, system_prompt: str, registry: ToolRegistry | None = None,
                 model: str | None = None, max_iterations: int = 8,
                 verbose: bool = False,
                 episodic_memory: Memory | None = None,
                 semantic_memory: Memory | None = None):
        self.system_prompt = system_prompt
        self.registry = registry or ToolRegistry()
        self.model = model
        # Default 8: enough for a realistic chain (look up, compute, look up
        # again, answer) with headroom, low enough to cap a stuck agent fast.
        self.max_iterations = max_iterations
        self.verbose = verbose
        # Optional long-term memory. Without these the agent is amnesiac: each
        # run() starts fresh. With them it carries continuity across runs.
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.messages: list[dict] = []
        # System prompt actually used this run (base prompt + injected memory).
        self._current_system_prompt = system_prompt

    # --- public API -------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Handle one user request, looping through tool calls until an answer."""
        # Pull relevant long-term memory into this run's system prompt first.
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
            # Loop budget exhausted — don't crash or return nothing.
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
        """One API call. Returns the assistant message object."""
        tools = self.registry.openai_schemas() if use_tools else None
        kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        resp = llm.chat(self._payload(), model=self.model, **kwargs)
        return resp.choices[0].message

    def _handle_tool_calls(self, message) -> None:
        """Run every tool the model requested and append the results."""
        # The assistant's tool-call turn must be in history before the results.
        self.messages.append(message.model_dump())
        for call in message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            if self.verbose:
                print(f"  [tool] {name}({args})")
            result = self.registry.dispatch(name, args)
            if self.verbose:
                print(f"  [ -> ] {result[:200]}")
            self.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    def _record_final(self, content: str) -> str:
        """Store and return the model's final text answer."""
        self.messages.append({"role": "assistant", "content": content})
        return content

    def _summarize_and_stop(self) -> str:
        """Reached max_iterations: ask for a best-effort answer, tools off.

        Better than crashing or returning "": the user gets whatever the agent
        managed to figure out, with a note that it stopped early.
        """
        self.messages.append({
            "role": "user",
            "content": (
                "You have reached the tool-call limit. Stop calling tools and "
                "answer now with whatever you have figured out so far."
            ),
        })
        message = self._call_llm(use_tools=False)
        answer = message.content or ""
        note = "[stopped after reaching max tool-call iterations] "
        return self._record_final(note + answer)

    # --- long-term memory -------------------------------------------------

    def _augment_system_prompt(self, user_input: str) -> str:
        """Prepend relevant recalled memories to the base system prompt.

        Episodic recall is recency-based; semantic recall is similarity-based.
        We inject only the top few of each so we add useful context without
        flooding the prompt (and the token bill) with noise.
        """
        blocks: list[str] = []
        if self.episodic_memory:
            recent = self.episodic_memory.retrieve(user_input, k=5)
            if recent:
                blocks.append("Recent conversation summaries:\n" + "\n".join(recent))
        if self.semantic_memory:
            facts = self.semantic_memory.retrieve(user_input, k=3)
            if facts:
                blocks.append("Relevant facts you have learned:\n" +
                              "\n".join(f"- {f}" for f in facts))

        if not blocks:
            return self.system_prompt
        memory_section = "\n\n## Memory (background context)\n" + "\n\n".join(blocks)
        return self.system_prompt + memory_section

    def _persist_memory(self, user_input: str, answer: str) -> None:
        """After a run, write an episodic summary and any semantic facts."""
        if self.episodic_memory:
            summary = self._summarize_run(user_input, answer)
            self.episodic_memory.add(summary, {"user_query": user_input})
            if self.verbose:
                print(f"  [episodic saved] {summary}")
        if self.semantic_memory:
            for fact in self._extract_facts(user_input, answer):
                self.semantic_memory.add(fact, {"source": "conversation"})
                if self.verbose:
                    print(f"  [semantic saved] {fact}")

    def _summarize_run(self, user_input: str, answer: str) -> str:
        """Ask the model for a 2-3 sentence summary of what just happened."""
        resp = llm.chat(
            [
                {"role": "system", "content": "Summarize the exchange in 2-3 sentences, "
                 "third person, factual. No preamble."},
                {"role": "user", "content": f"User asked: {user_input}\n\nAgent answered: {answer}"},
            ],
            model=self.model,
        )
        return resp.choices[0].message.content.strip()

    def _extract_facts(self, user_input: str, answer: str) -> list[str]:
        """Pull durable, reusable facts worth remembering (may be empty).

        We only want lasting knowledge (preferences, stable facts), not ephemera
        like "the weather last Tuesday". The model returns one fact per line, or
        the literal NONE when nothing is worth keeping.
        """
        resp = llm.chat(
            [
                {"role": "system", "content": (
                    "Extract durable facts worth remembering long-term (user preferences, "
                    "stable knowledge). Ignore ephemeral or time-bound details. Output one "
                    "fact per line, no bullets. If nothing is worth keeping, output exactly: NONE"
                )},
                {"role": "user", "content": f"User: {user_input}\n\nAgent: {answer}"},
            ],
            model=self.model,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.upper() == "NONE":
            return []
        return [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
