import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pageindex import page_index
from pageindex.utils import ChatGPT_API, extract_json, structure_to_list


def tree_search(question, structure, model):
    compact_tree = [{"node_id": n.get("node_id"), "title": n.get("title")} for n in structure_to_list(structure)]
    prompt = f"""
You are a retrieval planner.
Given a user question and a PageIndex tree, return the node_ids that are most relevant.

Question: {question}

Tree nodes:
{json.dumps(compact_tree, ensure_ascii=False)}

Reply in JSON:
{{
  "thinking": "short reasoning",
  "node_ids": ["0001", "0002"]
}}
Return JSON only.
"""
    result = extract_json(ChatGPT_API(model=model, prompt=prompt))
    return result.get("node_ids", [])


def answer_with_context(question, selected_nodes, model):
    context = [
        {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "text": node.get("text", ""),
        }
        for node in selected_nodes
    ]

    prompt = f"""
Answer the question using only the retrieved context below.
If context is insufficient, explicitly say so.

Question: {question}

Retrieved context:
{json.dumps(context, ensure_ascii=False)}
"""
    return ChatGPT_API(model=model, prompt=prompt)


def main():
    parser = argparse.ArgumentParser(description="OpenRouter demo for PageIndex")
    parser.add_argument("--pdf_path", required=True, help="Path to PDF")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument(
        "--model",
        default="moonshotai/kimi-k2",
        help="Model served through OpenRouter, e.g. moonshotai/kimi-k2",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    print("Building PageIndex tree...")
    indexed = page_index(
        args.pdf_path,
        model=args.model,
        if_add_node_id="yes",
        if_add_node_text="yes",
        if_add_node_summary="no",
        if_add_doc_description="no",
    )

    structure = indexed["structure"]
    all_nodes = structure_to_list(structure)

    print("Running tree search...")
    selected_node_ids = tree_search(args.question, structure, args.model)
    selected_nodes = [n for n in all_nodes if n.get("node_id") in set(selected_node_ids)]

    print("\nSelected node IDs:", selected_node_ids)
    print("\nGenerating answer...")
    answer = answer_with_context(args.question, selected_nodes, args.model)

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
