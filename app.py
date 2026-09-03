import json
import requests
import joblib
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration
# -----------------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
TOP_RESULTS = 30
EMBEDDINGS_FILE = "embeddings.joblib"


# -----------------------------
# Load the existing RAG index
# -----------------------------
@st.cache_resource
def load_embeddings():
    return joblib.load(EMBEDDINGS_FILE)


def create_embedding(text_list):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text_list,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    if "embeddings" not in data:
        raise RuntimeError(f"Ollama embedding error: {data}")

    return data["embeddings"]


def generate_answer(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Ollama generation error: {data}")

    return data["response"]


def retrieve_context(query, df, top_k=TOP_RESULTS):
    question_embedding = create_embedding([query])[0]

    all_embeddings = np.vstack(df["embedding"].values)

    similarities = cosine_similarity(
        all_embeddings,
        [question_embedding],
    ).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results


def build_prompt(query, results):
    context = results[
        ["title", "number", "start", "end", "text"]
    ].to_json(orient="records")

    return f"""
I am teaching web development in my Sigma web development course.

Here are relevant video subtitle chunks. Each chunk contains:
- video title
- video number
- start time in seconds
- end time in seconds
- subtitle text

CONTEXT:
{context}

USER QUESTION:
"{query}"

Instructions:
1. Answer the user's question using the provided course context.
2. Explain the answer in a natural and helpful way.
3. Clearly mention which video covers the topic.
4. Include the approximate timestamp (start/end) where the topic is taught.
5. Guide the learner to watch that part of the video.
6. Do not mention the internal JSON format, embeddings, cosine similarity, or retrieval process.
7. If the question is unrelated to the course, say that you can only answer questions related to the course.
8. Do not invent a video title, timestamp, or course information that is not supported by the context.
"""


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="RAG AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 RAG-Based AI Teaching Assistant")
st.caption(
    "Ask questions about the Sigma Web Development course and "
    "get answers grounded in the course content."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**Embedding model:** `{EMBED_MODEL}`")
    st.write(f"**LLM:** `{LLM_MODEL}`")
    st.write(f"**Retrieved chunks:** `{TOP_RESULTS}`")

    st.divider()
    st.markdown(
        "**Pipeline**\n"
        "1. User question\n"
        "2. BGE-M3 embedding\n"
        "3. Cosine-similarity retrieval\n"
        "4. Top relevant subtitle chunks\n"
        "5. Llama 3.2 answer generation"
    )

# Check required files
try:
    df = load_embeddings()
except FileNotFoundError:
    st.error(
        f"`{EMBEDDINGS_FILE}` was not found. "
        "Run `read_chunks.py` first to create the RAG index."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the RAG index: {exc}")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the course...")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the course and generating an answer..."):
            try:
                retrieved = retrieve_context(query, df)
                prompt = build_prompt(query, retrieved)
                answer = generate_answer(prompt)

                st.markdown(answer)

                with st.expander("🔎 Retrieved course sections"):
                    display_columns = [
                        "number",
                        "title",
                        "start",
                        "end",
                        "similarity",
                    ]

                    preview = retrieved[display_columns].copy()
                    preview["start"] = preview["start"].round(1)
                    preview["end"] = preview["end"].round(1)
                    preview["similarity"] = preview["similarity"].round(3)

                    st.dataframe(
                        preview,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to Ollama. Make sure Ollama is running "
                    "at http://localhost:11434."
                )
            except requests.exceptions.HTTPError as exc:
                st.error(f"Ollama returned an HTTP error: {exc}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

# Footer
st.divider()
st.caption(
    "Built with Python, Streamlit, Ollama, BGE-M3 embeddings, "
    "scikit-learn cosine similarity, and Llama 3.2."
)
