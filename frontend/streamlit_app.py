"""Interface de test Streamlit : sélecteur de personnage + chat streaming + sources RAG.

Usage : uv run --group ui streamlit run frontend/streamlit_app.py
(le backend doit tourner : uv run --group rag uvicorn backend.main:app --port 8080)
"""

import json
import os

import httpx
import streamlit as st

API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8080")
API_KEY = os.environ.get("CHATBOT_API_KEY", "change-me")

st.set_page_config(page_title="PulpFiction Chatbot", page_icon="🎬")
st.title("🎬 PulpFiction Chatbot — interface de test")


@st.cache_data(ttl=300)
def load_characters() -> list[dict]:
    r = httpx.get(f"{API_URL}/characters", timeout=10)
    r.raise_for_status()
    return r.json()

chars = load_characters()
char_by_name = {c["display_name"]: c["key"] for c in chars}
selected = st.sidebar.selectbox("Personnage", list(char_by_name))
character_key = char_by_name[selected]
top_k = st.sidebar.slider("Chunks RAG (top-k)", 0, 10, 4)
st.sidebar.caption("⏳ Premier message : le GPU Modal peut mettre 1-2 min à se réveiller.")

if "history" not in st.session_state or st.session_state.get("character") != character_key:
    st.session_state.history = []
    st.session_state.character = character_key

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if prompt := st.chat_input(f"Parle à {selected}…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        sources: list[dict] = []
        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/chat",
                headers={"X-API-Key": API_KEY},
                json={
                    "character": character_key,
                    "message": prompt,
                    "history": st.session_state.history[-10:],
                    "top_k": top_k,
                },
                timeout=240,
            ) as resp:
                resp.raise_for_status()
                event = None
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        if event == "sources":
                            sources = json.loads(data)
                            event = None
                        else:
                            answer += json.loads(data)["token"]
                            placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
        except httpx.HTTPError as e:
            st.error(f"Erreur API : {e}")

        if sources:
            with st.expander("📚 Sources RAG"):
                for s in sources:
                    st.write(f"`{s['score']:.3f}` — {s['source']} · {s['title']}")

    st.session_state.history += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
