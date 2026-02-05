"""Simple script to test embeddings against Supabase using Ollama/Qwen.

It sends a POST to the embeddings server (default http://localhost:11434/embeddings)
with `model` and `input` and expects an OpenAI-like response containing
`data[0].embedding`.
"""

import os
import textwrap
import requests
from supabase import create_client, Client
from dotenv import load_dotenv, find_dotenv

# ---- Load env
load_dotenv(find_dotenv(usecwd=True))
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Ollama embeddings endpoint and model
EMBEDDING_API_URL = os.environ.get(
    "EMBEDDING_API_URL", "http://localhost:11434/api/embeddings"
)
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

PDF_PATH = "human-nutrition-text.pdf"  # used as a filter in metadata
TOP_K = 3

queries = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
    "What are micronutrients?",
]


def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    for q in queries:
        # embed query via Ollama / embeddings endpoint
        r = requests.post(EMBEDDING_API_URL, json={"model": EMBED_MODEL, "prompt": q})
        if not r.ok:
            raise SystemExit(f"Embedding API error: {r.status_code} {r.text}")
        data = r.json()
        # Handle common OpenAI-like response shapes
        if data.get("data") and isinstance(data["data"], list) and data["data"]:
            e = data["data"][0].get("embedding")
        elif data.get("embedding"):
            e = data.get("embedding")
        else:
            raise SystemExit(f"Unexpected embedding response: {data}")

        # call your RPC with a metadata filter to this PDF
        resp = sb.rpc(
            "match_documents",
            {
                "query_embedding": e,
                "match_count": TOP_K,
                "filter": {"source": PDF_PATH},
            },
        ).execute()

        rows = resp.data or []
        print("\n" + "=" * 90)
        print(f"QUERY: {q}")
        if not rows:
            print("  (no matches)")
            continue

        for rank, r in enumerate(rows, start=1):
            page = (r.get("metadata") or {}).get("page", "?")
            sim = r.get("similarity", None)
            sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "?"
            preview = textwrap.shorten(
                r.get("content", "").replace("\n", " "), width=160
            )
            print(
                f"  [{rank}] page {page}  sim={sim_str}  chunk_index={r.get('chunk_index')}"
            )
            print(f"      {preview}")


if __name__ == "__main__":
    main()
