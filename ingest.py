# ingest.py
# pip install pymupdf supabase openai tqdm python-dotenv langchain

import os, uuid, re
import fitz  # PyMuPDF
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Load environment
load_dotenv(find_dotenv(usecwd=True))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# ---- Config
PDF_PATH = "human-nutrition-text.pdf"
DOC_ID = "nutrition-v1"  # keep this STABLE to avoid duplicates
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # 1024 dims
BATCH_EMBED = 100
BATCH_INSERT = 200

# Recursive chunking params
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 100  # character overlap between chunks

tokenizer = AutoTokenizer.from_pretrained(
    EMBED_MODEL
)  # matches Qwen embeddings tokenizer


def clean_text(t: str) -> str:
    # normalize whitespace and fix hyphenation across line breaks
    t = t.replace("\r", " ")
    t = re.sub(r"-\s*\n\s*", "", t)  # join "nutri-\n tion" => "nutrition"
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = t.replace("\n", " ").strip()
    return t


def pdf_pages(path: str):
    """Yield (page_number_1based, cleaned_text)."""
    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            txt = doc[i].get_text("text") or ""
            yield (i + 1, clean_text(txt))
    finally:
        doc.close()


def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    model = SentenceTransformer(EMBED_MODEL)

    # Initialize LangChain's recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # recursive splitting strategy
        length_function=len,
    )

    # Optional: keep the table clean for this document
    sb.table("chunks").delete().eq("doc_id", DOC_ID).execute()

    print("Reading PDF by pages...")
    pages = list(pdf_pages(PDF_PATH))

    # Build chunks with page metadata using LangChain's recursive splitter
    inputs, metas = [], []
    print(
        f"Chunking with recursive strategy (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})..."
    )
    for page, text in pages:
        if not text:
            continue
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            if chunk.strip():  # skip empty chunks
                inputs.append(chunk)
                metas.append({"page": page, "source": PDF_PATH})

    print(f"Built {len(inputs)} chunks from {PDF_PATH}")

    # Generate embeddings
    vectors = []
    print("Generating embeddings...")
    for i in tqdm(range(0, len(inputs), BATCH_EMBED), desc="Embedding"):
        batch = inputs[i : i + BATCH_EMBED]
        embeddings = model.encode(batch, convert_to_tensor=False)
        vectors.extend(embeddings.tolist())

    # Prepare rows
    rows = []
    for idx, (content, emb, meta) in enumerate(zip(inputs, vectors, metas)):
        rows.append(
            {
                "doc_id": DOC_ID,
                "chunk_index": idx,
                "content": content,
                "metadata": meta,  # contains {source, page}
                "embedding": emb,
            }
        )

    print("Uploading to Supabase...")
    for j in tqdm(range(0, len(rows), BATCH_INSERT), desc="Uploading"):
        sb.table("chunks").insert(rows[j : j + BATCH_INSERT]).execute()

    print(f"🎉 Done! Inserted {len(rows)} chunks for doc_id={DOC_ID}")


if __name__ == "__main__":
    main()
