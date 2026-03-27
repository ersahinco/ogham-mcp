# Ogham MCP — Embedding System

## Provider Architecture

Four embedding providers, selected via `EMBEDDING_PROVIDER` env var:

| Provider | Model | Default Dim | Batch Limit | Notes |
|----------|-------|-------------|-------------|-------|
| `ollama` (default) | `embeddinggemma` | 512 | 10 | Local, free, dimension control via `dimensions` param |
| `openai` | `text-embedding-3-small` | 1024 | 500 | Supports dimension reduction |
| `mistral` | `mistral-embed` | 1024 | 32 | 16,384 token limit per request |
| `voyage` | `voyage-4-lite` | 1024 | 500 | 1,000 inputs per request, auto-batched |

Each provider has singleton client instances (lazy-created). Dimension validation runs after every embedding call.

## Caching (`embedding_cache.py`)

### Architecture
- **SQLite-backed** persistent cache at `~/.cache/ogham/embeddings.db`
- Thread-safe via `threading.Lock`
- Schema: `key TEXT PRIMARY KEY, value BLOB, created_at REAL`

### Cache Key
```python
SHA256(f"{provider}:{dim}:{text}")
```
Switching providers or dimensions automatically invalidates cached vectors because the key prefix changes.

### Eviction
- LRU-style: when size exceeds `max_size` (default 10,000), deletes oldest entries by `created_at`
- Eviction runs after every `put()`

### Operations
- `get(key)` → cache hit/miss tracking
- `put(key, embedding)` → insert/replace + evict
- `clear()` → wipe all + reset stats
- `stats()` → size, max_size, hits, misses, hit_rate

## Batch Embedding

`generate_embeddings_batch(texts, batch_size?, on_progress?)`:
1. Check cache for each text
2. Group uncached texts into batches (size from `EMBEDDING_BATCH_SIZE` or provider default)
3. Call provider batch API
4. Cache results
5. Return in original order

Progress callback `on_progress(embedded_so_far, total)` fires after each batch.

## Retry

All provider calls (single + batch) are wrapped with:
```python
@with_retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, OSError))
```

Exponential backoff: 0.5s → 1.0s → 2.0s

The retry decorator also catches `psycopg.OperationalError` if psycopg is installed (for database retry).
