from velorag import VeloRAG, Settings

rag = VeloRAG()

# Ingest a sample document
chunks = rag.ingest_file_sync("README.md") if __import__("pathlib").Path("README.md").exists() else []

# Or ingest text directly
rag.ingest_file_sync("sample.txt") if __import__("pathlib").Path("sample.txt").exists() else None

# Search
results = rag.search("What is VeloRAG?", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] ({r.retrieval_source}) {r.chunk.text[:100]}...")
