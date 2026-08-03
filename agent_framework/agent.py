"""The Agent — a ReAct loop around the LLM: reason, call a tool, observe, repeat
until it produces a text answer."""

from __future__ import annotations

from . import llm
from .tools import ToolRegistry


class Agent:
    def __init__(self, system_prompt: str, registry: ToolRegistry | None = None,
                 model: str | None = None, max_iterations: int = 8, verbose: bool = False):
        self.system_prompt = system_prompt
        self.registry = registry or ToolRegistry()
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages: list[dict] = []

    def run(self, user_input: str) -> str:
        """Handle one request, looping through tool calls until an answer."""
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(self.max_iterations):
            message = self._call_llm()
            if message.tool_calls:
                self._handle_tool_calls(message)
                continue
            return self._record_final(message.content)
        return self._summarize_and_stop()

    def reset(self) -> None:
        """Forget the current conversation (keeps tools and system prompt)."""
        self.messages = []

    # --- loop steps -------------------------------------------------------

    def _payload(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

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
