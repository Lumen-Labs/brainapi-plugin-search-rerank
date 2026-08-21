# search-rerank

Optional second-stage **cross-encoder** for BrainAPI `POST /retrieve/search`. It registers `rerank=plugin:cross-encoder`. Core hybrid search (BM25 + dense + filters) works if this plugin is absent. It does **not** run on `/retrieve/context`.

| | |
|---|---|
| Registry name | `search-rerank` |
| Version | `0.1.0` |
| BrainAPI | `>=2.17.0` |
| Hook | `rerank=plugin:cross-encoder` |
| Default model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Health | `GET /search-rerank/health` |

Unknown or missing `plugin:<name>` is **400**, never a silent no-op that looks like a ranking miss.

## Install

```bash
git clone https://github.com/Lumen-Labs/brainapi-plugin-search-rerank.git plugins/search-rerank
```

Or:

```bash
./bin/brainapi install search-rerank
```

Restart the API. First rerank call lazy-loads `sentence_transformers.CrossEncoder` (install `sentence-transformers` in the BrainAPI environment if it is not already there).

## Quick start

```bash
curl -s "$BRAINAPI_URL/search-rerank/health" -H "BrainPAT: $BRAINPAT_TOKEN"

curl -X POST "$BRAINAPI_URL/retrieve/search" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -H "X-Brain-ID: searchbenchsmoke" \
  -d '{
    "query": "navy wool coat",
    "k": 10,
    "rerank": "plugin:cross-encoder"
  }'
```

Benchmark harness: `--rerank plugin:cross-encoder`.

## How it ranks

1. Core (or another first stage) returns a candidate list with `text`.
2. This plugin scores `(query, text)` pairs with a cross-encoder.
3. Candidates are sorted by score descending and cut to `k`.

Core caps how many candidates are reranked:

| Search `mode` | Retrieve pool | Rerank cap |
|---|---|---|
| `default` (omitted) | request `k` | `RERANK_MAX_K = 10` |
| `catalog` | `min(200, max(k, 50))` | `CATALOG_RERANK_MAX_K = 50` |

Health reports `"max_k": 10` for the default path. Catalog mode is the deeper, slower path — not the ADR-007 ~200 ms default.

### 4-class ESCI models

If the model returns 4 logits per pair, scores are a softmax-weighted gain `(1.0, 0.1, 0.01, 0.0)` (Exact / Substitute / Complement / Irrelevant). Binary / single-logit models use the raw score (or the first logit).

## Configuration

| Env | Default |
|---|---|
| `SEARCH_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

Tests can inject `set_predict(fn)` instead of loading Hugging Face weights.

## API

### `GET /search-rerank/health`

```json
{
  "plugin": "search-rerank",
  "rerank": "plugin:cross-encoder",
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "loaded": false,
  "max_k": 10,
  "error": null
}
```

`loaded` becomes `true` after the first successful predict. `error` is set if the last load failed.

There is no index to build — this plugin only reranks lists produced by `/retrieve/search`.

## Layout

```text
search-rerank/
  plugin.yaml
  main.py       # register_search_reranker("cross-encoder", …)
  rerank.py     # CrossEncoder + ESCI gains
  routes.py     # GET /search-rerank/health
```

## Publishing

Pushes to `main` publish to the BrainAPI registry via GitHub Actions.

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).

## Related

- [search-splade](https://github.com/Lumen-Labs/brainapi-plugin-search-splade) — learned-sparse first stage
- [search-colbert](https://github.com/Lumen-Labs/brainapi-plugin-search-colbert) — late-interaction first stage
- [BrainAPI](https://github.com/Lumen-Labs/brainapi2)
- `docs/research/18-search-eval-protocol.md` on brainapi2
