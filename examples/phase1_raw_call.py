"""Phase 1: talk to an LLM programmatically.

The simplest possible thing: a hardcoded prompt, one API call, and a dump of the
*entire* response object — not just the text. Run it and read what comes back.

    python examples/phase1_raw_call.py

Key ideas to notice in the output:
- The API takes a `messages` array of {role, content}, not a single string.
- `roles` are system / user / assistant.
- The API is STATELESS: it returns only the next message. Any conversation
  history must be kept by *us* (that's Phase 2).
- The `usage` field reports prompt/completion/total tokens — this is what you pay
  for. A token is a chunk of text (~4 chars of English), not a word.
- `choices[0].finish_reason` says why generation stopped ("stop", "length", ...).
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

messages = [
    {"role": "system", "content": "You are a concise, helpful assistant."},
    {"role": "user", "content": "What is the capital of France? Answer in one word."},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("=== FULL RESPONSE OBJECT ===")
print(response.model_dump_json(indent=2))

print("\n=== JUST THE TEXT ===")
print(response.choices[0].message.content)

print("\n=== TOKEN USAGE ===")
usage = response.usage
print(f"prompt_tokens     = {usage.prompt_tokens}")
print(f"completion_tokens = {usage.completion_tokens}")
print(f"total_tokens      = {usage.total_tokens}")

print("\n=== FINISH REASON ===")
print(response.choices[0].finish_reason)
