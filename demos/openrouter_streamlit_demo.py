import json
import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pageindex import page_index
from pageindex.utils import ChatGPT_API, extract_json, structure_to_list


def tree_search(question, structure, model):
    compact_tree = [
        {
            "node_id": n.get("node_id"),
            "title": n.get("title"),
            "summary": n.get("summary", n.get("prefix_summary", "")),
        }
        for n in structure_to_list(structure)
    ]

    prompt = f"""
You are a retrieval planner for long documents.
Select the most relevant node_ids for answering the user's question.

Question: {question}

Tree nodes (id/title/summary):
{json.dumps(compact_tree, ensure_ascii=False)}

Reply in JSON only:
{{
  "thinking": "short reasoning",
  "node_ids": ["0001", "0002"]
}}
"""

    result = extract_json(ChatGPT_API(model=model, prompt=prompt))
    return result.get("node_ids", [])


def answer_with_context(question, selected_nodes, model):
    context = [
        {
            "node_id": n.get("node_id"),
            "title": n.get("title"),
            "text": n.get("text", ""),
        }
        for n in selected_nodes
    ]

    prompt = f"""
Answer the question using only the provided context.
If context is insufficient, clearly say what is missing.

Question: {question}

Context:
{json.dumps(context, ensure_ascii=False)}
"""
    return ChatGPT_API(model=model, prompt=prompt)


def save_uploaded_file(uploaded_file):
    demos_tmp_dir = REPO_ROOT / "demos" / "tmp"
    demos_tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = demos_tmp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)


def ensure_openrouter_env():
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("OPENROUTER_API_KEY is missing. Copy .env.example to .env and set your key.")
        st.stop()


def main():
    st.set_page_config(page_title="PageIndex + OpenRouter Demo", layout="wide")
    st.title("📄 PageIndex Demo (OpenRouter + Kimi)")
    st.caption("Upload a PDF, build the PageIndex tree, then ask a question.")

    ensure_openrouter_env()

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Model", value="moonshotai/kimi-k2")
        toc_check_pages = st.number_input("TOC check pages", min_value=1, max_value=100, value=20)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if "indexed" not in st.session_state:
        st.session_state.indexed = None

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("1) Build PageIndex Tree", disabled=uploaded_file is None):
            pdf_path = save_uploaded_file(uploaded_file)
            with st.spinner("Indexing PDF with PageIndex... this may take a while"):
                st.session_state.indexed = page_index(
                    pdf_path,
                    model=model,
                    toc_check_page_num=toc_check_pages,
                    if_add_node_id="yes",
                    if_add_node_text="yes",
                    if_add_node_summary="yes",
                    if_add_doc_description="no",
                )
            st.success("Tree built successfully.")

    with col2:
        question = st.text_input("2) Ask a question", placeholder="What is the main argument in this document?")
        ask_disabled = st.session_state.indexed is None or not question.strip()
        if st.button("3) Retrieve + Answer", disabled=ask_disabled):
            structure = st.session_state.indexed["structure"]
            all_nodes = structure_to_list(structure)

            with st.spinner("Selecting relevant nodes..."):
                selected_ids = tree_search(question, structure, model)
            selected_nodes = [n for n in all_nodes if n.get("node_id") in set(selected_ids)]

            with st.spinner("Generating answer..."):
                answer = answer_with_context(question, selected_nodes, model)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved node IDs")
            st.code(json.dumps(selected_ids, indent=2, ensure_ascii=False), language="json")

            with st.expander("Retrieved node details"):
                st.json(selected_nodes)

    if st.session_state.indexed:
        st.subheader("Tree preview")
        st.json(st.session_state.indexed["structure"])


if __name__ == "__main__":
    main()
