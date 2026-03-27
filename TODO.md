# BEAM Benchmark: Neural Sparse vs tsvector

Goal: Run ogham's BEAM benchmark comparing current tsvector hybrid search against BGE-M3 neural sparse vectors, focusing on event_ordering and summarization categories (the weak spots Kevin identified). Produce per-category metrics to demonstrate whether the switch is worth the complexity.

## Prerequisites

- [ ] **Clone BEAM dataset**
  Clone https://github.com/mohammadtavakoli78/BEAM alongside the ogham repo.

- [ ] **Set up benchmarks/.env.local**
  Configure DATABASE_URL, EMBEDDING_PROVIDER, etc. for local benchmark runs.
  Current default uses Voyage — we'll need to decide whether to benchmark with Ollama (dense) or run FlagEmbedding for the sparse path.

- [ ] **Install FlagEmbedding**
  `uv add FlagEmbedding` (or as a dev dependency). This pulls PyTorch + transformers.
  Alternatively, test the ONNX path (`yuniko-software/bge-m3-onnx`) for lighter weight.

## Phase 1: Baseline — current tsvector hybrid search

- [ ] **Ingest 100K bucket**
  `uv run python3 benchmarks/beam_benchmark.py --ingest --bucket 100K`

- [ ] **Run baseline eval (all categories)**
  `uv run python3 benchmarks/beam_benchmark.py --eval --bucket 100K`
  Save results to `benchmarks/beam_results/` for comparison.

- [ ] **Note baseline scores for event_ordering and summarization**
  These are the categories Kevin wants to see improvement on.

## Phase 2: Add neural sparse embedding path

- [ ] **Add FlagEmbedding provider to embeddings.py**
  New provider option (e.g. `EMBEDDING_PROVIDER=flagembedding`) that calls BGEM3FlagModel.encode() and returns dense + sparse + optionally ColBERT vectors.
  Keep backward compat — Ollama/OpenAI/Voyage continue returning dense only.

- [ ] **Add sparse_embedding column to schema**
  New column on memories table. pgvector `sparsevec` if non-zero count stays under 1,000 for benchmark data; otherwise investigate VectorChord `svector`.
  Write a migration script.

- [ ] **Store sparse vectors during ingestion**
  Modify store_memory / store_memories_batch to accept and persist sparse embeddings when the provider supplies them.

- [ ] **Create hybrid_search_memories_sparse() SQL function**
  New variant that fuses: `0.7 × dense_cosine + 0.3 × sparse_dot_product` instead of `0.7 × dense_cosine + 0.3 × tsvector_rank`.
  Keep the original function intact for A/B comparison.

- [ ] **Wire sparse search into service.py**
  Add a config flag or search parameter to select sparse vs tsvector hybrid mode.
  search_memories_enriched() should route to the appropriate SQL function.

## Phase 3: Benchmark comparison

- [ ] **Re-ingest 100K bucket with sparse embeddings**
  Same data, but now each memory also has a sparse_embedding column populated.

- [ ] **Run eval with sparse hybrid search**
  Same eval harness, but search_memories_enriched() now uses the sparse path.
  Save results separately.

- [ ] **Compare per-category metrics**
  Diff baseline vs sparse for Recall@5/10/20, MRR, NDCG@10.
  Focus on event_ordering and summarization.
  Document whether the improvement justifies the added complexity.

- [ ] **Share results on issue #13**
  Post the comparison table as a comment on ogham-mcp/ogham-mcp#13.

## Phase 4 (stretch): ColBERT reranking

- [ ] **Add ColBERT vector storage**
  vector[] column or separate token embeddings table.
  Only worth pursuing if Phase 3 shows neural sparse is a clear win.

- [ ] **Implement MaxSim reranker**
  Rerank top-K results from dense+sparse using ColBERT MaxSim.
  Client-side first (no VectorChord dependency), measure improvement.

- [ ] **Benchmark ColBERT reranking**
  Same eval harness, compare dense+sparse vs dense+sparse+ColBERT rerank.

## Open questions

- What size is the sparse output for typical BEAM conversation rounds? Need to check whether pgvector's 1,000 non-zero limit is a blocker or if the data fits.
- Is FlagEmbedding's memory footprint acceptable alongside Ollama? Or should we stop Ollama and use FlagEmbedding for dense too?
- Should the benchmark use the 100K bucket only (fast iteration) or run 500K/1M as well for more statistical power?
