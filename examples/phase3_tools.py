"""Phase 3 demo: one full tool-calling round trip, done by hand.

    python examples/phase3_tools.py

This is the mechanism the Agent loop (Phase 4) will automate. Steps:
  1. Send messages + tool schemas to the API.
  2. If the reply has `tool_calls`, run each via the registry.
  3. Append the assistant's tool_call message and a `tool` result message.
  4. Call the API again so the model can answer using the result.
"""

import json

from agent_framework import llm
from agent_framework.builtin_tools import default_registry


def main() -> None:
    registry = default_registry()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "How many feet is 3.5 kilometers?"},
    ]

    # --- round 1: model decides to call a tool ---
    resp = llm.chat(messages, tools=registry.openai_schemas(), tool_choice="auto")
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print("Model answered directly:", msg.content)
        return

    # Record the assistant's tool-call turn verbatim (the API requires this).
    messages.append(msg.model_dump())

    for call in msg.tool_calls:
        name = call.function.name
        args = call.function.arguments  # JSON string
        print(f"-> model wants: {name}({args})")
        result = registry.dispatch(name, args)
        print(f"<- tool result: {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

    # --- round 2: model answers using the tool result ---
    final = llm.chat(messages, tools=registry.openai_schemas())
    print("\nFINAL:", final.choices[0].message.content)


if __name__ == "__main__":
    main()
