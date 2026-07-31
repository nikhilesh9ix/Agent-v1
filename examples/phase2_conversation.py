"""Phase 2 demo: a multi-turn conversation that remembers earlier turns.

    python examples/phase2_conversation.py

Watch it recall a fact stated three turns earlier. It works even though the API
is stateless because ConversationManager resends the whole history each call.
"""

from agent_framework.conversation import ConversationManager


def main() -> None:
    cm = ConversationManager("You are a concise, friendly assistant.")

    print("USER: My name is Alex. Remember that.")
    print("BOT :", cm.chat("My name is Alex. Remember that."))

    print("\nUSER: I like hiking and strong coffee.")
    print("BOT :", cm.chat("I like hiking and strong coffee."))

    print("\nUSER: What is my name and one thing I like?")
    print("BOT :", cm.chat("What is my name and one thing I like?"))

    print(f"\n[history has {len(cm.get_history())} messages]")


if __name__ == "__main__":
    main()
