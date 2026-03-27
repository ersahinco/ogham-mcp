I've been using ogham with BGE-M3 (via Ollama) and while digging into how the model works, I realized it produces three types of vectors but ogham only uses one (dense). The other two — sparse and ColBERT multi-vector — seem like they could meaningfully improve retrieval quality, and the model already knows how to generate them. I wanted to open a discussion about whether this is worth exploring and what the tradeoffs would be.

## What is BGE-M3?

For context in case others come across this: [BGE-M3](https://huggingface.co/BAAI/bge-m3) is an embedding model from BAAI designed around three properties — **M**ulti-lingual (100+ languages), **M**ulti-functionality (three vector types from one model), and **M**ulti-granularity (8,192 token context window).

It runs locally via Ollama (~1.2GB) and uses self-knowledge distillation, which makes its dense vectors competitive with much larger models. Ogham already supports it as an embedding provider — the interesting part is that we're only using one-third of what it can produce.

## The three vector types

| Type | Shape | What it captures |
|------|-------|-----------------|
| Dense | `(1024,)` per text | Single-vector semantic meaning — what ogham uses now |
| Sparse | `{token: weight}` dict, ~20-200 non-zero entries | Learned term importance — like BM25 but trained by the model |
| ColBERT | `(seq_len, 1024)` per text | Per-token vectors with MaxSim scoring for precise token-level matching |

The one that jumped out to me: ogham's hybrid search currently does `0.7 × dense_cosine + 0.3 × tsvector_keyword`, where the keyword side is PostgreSQL's built-in bag-of-words full-text search. BGE-M3's sparse vectors are a trained neural model that understands contextual term importance — it seems like a direct upgrade for that keyword slot, and the model is already producing the information internally.

ColBERT is the more exotic one. Instead of compressing a whole memory into a single vector, it keeps one vector per token and does fine-grained matching at query time. Probably overkill for many use cases, but interesting for precise factual recall.

## The Ollama limitation

One wrinkle: Ollama only exposes dense vectors from BGE-M3. There's an [open issue](https://github.com/ollama/ollama/issues/6230) (Aug 2024, 46+ upvotes, no movement) requesting sparse vector support. The GGUF quantization may have stripped the sparse/ColBERT heads entirely.

To get all three outputs, you'd need the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) Python library:

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output = model.encode(
    ['some text'],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)

dense   = output['dense_vecs']       # np.ndarray (N, 1024)
sparse  = output['lexical_weights']  # list of {token_str: weight} dicts
colbert = output['colbert_vecs']     # list of (seq_len, 1024) arrays
```

There's also an [ONNX export](https://github.com/yuniko-software/bge-m3-onnx) that avoids the PyTorch dependency. Either way, this means adding a new embedding provider path — not a trivial change.

## PostgreSQL storage: what's possible today

### Sparse vectors

pgvector added `sparsevec` in 0.7.0, but HNSW indexing caps at **1,000 non-zero elements** — BGE-M3 sparse output on longer text can exceed this.

[VectorChord](https://github.com/tensorchord/VectorChord) (successor to pgvecto.rs) has `svector` with no cardinality limit and better SIMD performance. It's a drop-in PostgreSQL extension.

### ColBERT multi-vectors

VectorChord 0.3+ added native MaxSim:

```sql
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embeddings vector(1024)[]   -- array of token vectors
);

CREATE INDEX ON items USING vchordrq (embeddings vector_maxsim_ops);

-- Query: MaxSim over query token vectors
SELECT id FROM items
ORDER BY embeddings @# ARRAY['[...]'::vector, '[...]'::vector]
LIMIT 10;
```

Benchmarked at ~35ms per query on 57K documents. Without VectorChord, ColBERT in plain pgvector would require a token-per-row table and a cross-join — only viable as a reranker over a small candidate set.

## One possible path

This is just how I'd think about phasing it, not necessarily a proposal:

**Sparse first**: swap `tsvector` for BGE-M3 sparse vectors in the hybrid search RRF fusion. This seems like the highest value-to-effort ratio — it slots into the existing architecture and the keyword side of hybrid search is the natural place for it.

**ColBERT later (maybe)**: add as a reranker over top-K dense+sparse results, or as a full retrieval mode via VectorChord. The storage cost is significant (one 1024-dim vector per token per memory), so this might only make sense for certain deployment sizes.

## Things I'm not sure about

- **Is this actually worth the complexity?** Ogham's current hybrid search works well. The tsvector keyword component is simple and fast. Is the quality improvement from neural sparse vectors worth adding FlagEmbedding as a dependency and a new provider path?
- **Dependency weight**: FlagEmbedding pulls in PyTorch + transformers. That's a big addition for a tool that currently runs lean with just Ollama. The ONNX path is lighter but less proven.
- **pgvector vs VectorChord**: Adding VectorChord as a dependency (or requirement for sparse/ColBERT) changes the deployment story. Is there a way to do this that degrades gracefully — e.g., use neural sparse if VectorChord is available, fall back to tsvector if not?
- **ColBERT storage at scale**: For a personal memory system with hundreds of memories this is fine. For larger deployments, the per-token storage could get expensive. Is there a natural cutoff where it stops making sense?
- **Other models**: This is framed around BGE-M3 specifically. Are there other multi-vector models worth considering, or is BGE-M3 the clear choice for this use case?

## Prior art

- Vespa: [full BGE-M3 three-mode tutorial](https://vespa-engine.github.io/pyvespa/examples/mother-of-all-embedding-models-cloud.html) — the most complete native implementation
- Qdrant: [BGE-M3 sample](https://github.com/yuniko-software/bge-m3-qdrant-sample) — dense+sparse with client-side ColBERT reranking
- Milvus: built-in `BGEM3EmbeddingFunction` for dense+sparse (ColBERT is an [open request](https://github.com/milvus-io/milvus/issues/31581))
- VectorChord: [MaxSim announcement](https://blog.vectorchord.ai/vectorchord-03-bringing-efficient-multi-vector-contextual-late-interaction-in-postgresql) — PostgreSQL-native ColBERT

Would love to hear your thoughts — whether this is on the roadmap, conflicts with other plans, or if there are tradeoffs I'm missing.
