"""ConversationManager — stateful multi-turn chat (no tools).

The API is stateless, so this holds the message history and resends it each call.
"""

from __future__ import annotations

from . import llm


class ConversationManager:
    def __init__(self, system_prompt: str = "You are a helpful assistant.",
                 model: str | None = None):
        self.system_prompt = system_prompt
        self.model = model
        self.messages: list[dict] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def get_response(self) -> str:
        """Send the full history, record and return the assistant reply."""
        payload = [{"role": "system", "content": self.system_prompt}, *self.messages]
        content = llm.chat(payload, model=self.model).choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        return content

    def chat(self, user_input: str) -> str:
        self.add_user_message(user_input)
        return self.get_response()

    def clear(self) -> None:
        """Reset the conversation, keep the system prompt."""
        self.messages = []

    def get_history(self) -> list[dict]:
        return self.messages
