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
from .tools import ToolRegistry


class Agent:
    """Runs the ReAct loop: reason -> act (tool) -> observe -> repeat -> answer."""

    def __init__(self, system_prompt: str, registry: ToolRegistry | None = None,
                 model: str | None = None, max_iterations: int = 8,
                 verbose: bool = False):
        self.system_prompt = system_prompt
        self.registry = registry or ToolRegistry()
        self.model = model
        # Default 8: enough for a realistic chain (look up, compute, look up
        # again, answer) with headroom, low enough to cap a stuck agent fast.
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages: list[dict] = []

    # --- public API -------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Handle one user request, looping through tool calls until an answer."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            message = self._call_llm()
            if message.tool_calls:
                self._handle_tool_calls(message)
                continue
            return self._record_final(message.content)

        # Loop budget exhausted — don't crash or return nothing.
        return self._summarize_and_stop()

    def reset(self) -> None:
        """Forget the current conversation (keeps tools and system prompt)."""
        self.messages = []

    # --- loop steps -------------------------------------------------------

    def _payload(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

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
