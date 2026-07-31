"""Streamlit chat frontend for the from-scratch agent framework.

    streamlit run app.py

A thin UI over agent_framework.Agent: pick a provider/model, edit the system
prompt, toggle tools and memory, and chat. Each assistant turn shows the tools
the agent actually called, so you can watch the ReAct loop work.
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from agent_framework import Agent, SQLiteMemory, llm
from agent_framework.builtin_tools import default_registry

load_dotenv()

st.set_page_config(page_title="Agent-v1", page_icon="🤖", layout="centered")


def available_providers() -> list[str]:
    """Providers whose API key is present in the environment."""
    out = []
    if os.getenv("GROQ_API_KEY"):
        out.append("groq")
    if os.getenv("OPENAI_API_KEY"):
        out.append("openai")
    return out


# Known models per provider; first entry is the default.
MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct",
    ],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
}


def build_agent(provider: str, model: str, system_prompt: str, use_tools: bool,
                use_episodic: bool, use_semantic: bool, max_iter: int) -> Agent:
    """Construct a fresh Agent for the chosen configuration."""
    llm.configure(provider=provider, model=model)
    registry = default_registry() if use_tools else None

    episodic = SQLiteMemory("streamlit_memory.db") if use_episodic else None
    semantic = None
    if use_semantic:
        try:
            from agent_framework import ChromaMemory
            semantic = ChromaMemory()
        except Exception as e:  # noqa: BLE001
            st.sidebar.warning(f"Semantic memory unavailable: {e}")

    return Agent(
        system_prompt=system_prompt,
        registry=registry,
        model=model,
        max_iterations=max_iter,
        episodic_memory=episodic,
        semantic_memory=semantic,
    )


def extract_tool_events(messages: list[dict], start: int) -> list[dict]:
    """Pull (name, args, result) for tool calls made since index `start`."""
    results = {m.get("tool_call_id"): m.get("content", "")
               for m in messages[start:] if m.get("role") == "tool"}
    return [
        {"name": c["function"]["name"], "args": c["function"].get("arguments", "{}"),
         "result": results.get(c["id"], "")}
        for m in messages[start:] if m.get("role") == "assistant" and m.get("tool_calls")
        for c in m["tool_calls"]
    ]


def render_tools(events: list[dict]) -> None:
    if not events:
        return
    with st.expander(f"🔧 {len(events)} tool call(s)"):
        for ev in events:
            st.markdown(f"**{ev['name']}**")
            st.code(f"args: {ev['args']}\nresult: {ev['result']}", language="text")


# --- sidebar: configuration ----------------------------------------------

providers = available_providers()
st.sidebar.title("⚙️ Configuration")

if not providers:
    st.sidebar.error("No API key found. Add GROQ_API_KEY or OPENAI_API_KEY to .env")
    st.stop()

provider = st.sidebar.selectbox("Provider", providers, index=0)
model = st.sidebar.selectbox("Model", MODELS[provider])
system_prompt = st.sidebar.text_area(
    "System prompt",
    value="You are a helpful assistant. Use tools when they improve accuracy.",
    height=120,
)
use_tools = st.sidebar.checkbox("Enable tools", value=True)
use_episodic = st.sidebar.checkbox("Episodic memory (SQLite)", value=False)
use_semantic = st.sidebar.checkbox(
    "Semantic memory (Chroma, needs OpenAI key)",
    value=False,
    disabled=not os.getenv("OPENAI_API_KEY"),
)
max_iter = st.sidebar.slider("Max tool iterations", 1, 15, 8)

if use_tools:
    st.sidebar.caption("Tools: " + ", ".join(default_registry().names()))

# Rebuild the agent (and reset the chat) whenever the configuration changes.
sig = (provider, model, system_prompt, use_tools, use_episodic, use_semantic, max_iter)
if st.session_state.get("sig") != sig:
    st.session_state.sig = sig
    st.session_state.agent = build_agent(*sig)
    st.session_state.history = []

if st.sidebar.button("🗑️ Reset conversation"):
    st.session_state.agent.reset()
    st.session_state.history = []

# --- main: chat ----------------------------------------------------------

st.title("🤖 Agent-v1")
st.caption(f"from-scratch agent framework · **{provider}** · `{model}`")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        render_tools(turn.get("tools", []))

prompt = st.chat_input("Ask the agent…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = st.session_state.agent
    start = len(agent.messages)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = agent.run(prompt)
            except Exception as e:  # noqa: BLE001
                answer = f"⚠️ Error: {type(e).__name__}: {e}"
        events = extract_tool_events(agent.messages, start)
        st.markdown(answer)
        render_tools(events)

    st.session_state.history.append({"role": "assistant", "content": answer, "tools": events})
