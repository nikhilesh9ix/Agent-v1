"""The ConversationManager.

The API is stateless — it forgets everything between calls. So *someone* has to
hold the running conversation and resend it every time. That someone is this
class. It bundles the state (system prompt + message list) with the behavior
(add messages, call the API), which is exactly what an object is for.

    cm = ConversationManager("You are a helpful assistant.")
    print(cm.chat("My name is Alex. Remember that."))
    print(cm.chat("What is my name?"))   # -> "Alex", because we resent the history
"""

from __future__ import annotations

from . import llm


class ConversationManager:
    """Holds a multi-turn conversation and talks to the LLM on its behalf."""

    def __init__(self, system_prompt: str = "You are a helpful assistant.",
                 model: str | None = None):
        self.system_prompt = system_prompt
        self.model = model
        # The message list is the conversation's memory. The system prompt is
        # kept separate so clear() can wipe the chat without losing the persona.
        self.messages: list[dict] = []

    def add_user_message(self, text: str) -> None:
        """Append a user turn to the history."""
        self.messages.append({"role": "user", "content": text})

    def _build_payload(self) -> list[dict]:
        """Full messages array we send to the API: system prompt + history."""
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

    def get_response(self) -> str:
        """Send the current history to the API and record the assistant reply.

        Because the API is stateless, we always send the entire conversation;
        the model reconstructs context purely from what we resend.
        """
        response = llm.chat(self._build_payload(), model=self.model)
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        return content

    def chat(self, user_input: str) -> str:
        """Convenience: add a user message and get the assistant's reply."""
        self.add_user_message(user_input)
        return self.get_response()

    def clear(self) -> None:
        """Reset the conversation but keep the system prompt/persona."""
        self.messages = []

    def get_history(self) -> list[dict]:
        """Return the raw message list (a live reference — copy if you mutate)."""
        return self.messages
