# Contract reference

This is a navigation reference to maintained definitions and validators. The
root [agent guide](../AGENTS.md) owns engineering rules; configuration and live
published metadata own runtime facts. Change a contract together with its
validator, consumer, compatibility policy, and regression coverage.

## Sources, grids, and row identity

- Config: [base](../configs/nys-georsclip.json) and
  [supplement](../configs/nys-georsclip-gapfill.json); field validation lives in
  `model.load_config`, native raster/grid validation in `tiling.py`.
- RGB is bands 1/2/3, native 0.6 m/pixel. Default windows are 256 x 256, stride
  192, full-size edge-anchored windows, at least 95% valid pixels.
- Global identity is `(dataset_id, shard_id, embedding_id)`; row number equals
  `embedding_id` inside one shard. Legacy two-part exclusions apply only to
  the base collection. Federated validation qualifies before searching.
- Persisted geometry is EPSG:4326 with longitude first. Original UTM CRS,
  transforms, and pixel windows remain separate. Display uses EPSG:3857.
- Release year and acquisition date are distinct. Valid base imagery takes
  display precedence; base-only display/search remain available.

## Persisted formats and authority

| Artifact | Schema/validator owner | Authority and compatibility |
| --- | --- | --- |
| Acquisition `manifest.json`, records, completion | `download_nys_naip.py`; `gapfill.load_inventory` | Source/provider identity, hashes, dates, original paths; preserve |
| Collection `run.json` | `embed_georsclip.build_recipe` | Recipe and `model_contract`, run fingerprint; source/dependency/code hashes |
| `collection.json` | `search_text.read_collection` | Published shard snapshot, aggregate counts, embedding dimension, state |
| Three-file shard | `bundles.METADATA_SCHEMA`, `metadata_table`, `validate_bundle` | Atomic vectors/GeoParquet/run publication; alignment and optional hash validation |
| `search-index/manifest.json` | `search_index.read_manifest`, `IndexReader` | Format version 1, IVF-PQ, exact collection snapshot and source offsets; checksummed native artifact |
| Frontend search JSON | `search_worker.run_request`, `retrieval.ExactRetrieval.text`, `Application.saved` | Query, dataset fingerprints, coverage snapshot, metadata, scores, diagnostics |
| Imported CLI search | `Application.saved_backend` | Must resolve identity against compatible published bundles |
| Cached text vectors | `ExactRetrieval.text` | Query + run-fingerprint key; finite normalized vector with matching dimension |
| Frontend bookmarks/state | `Application.save_bookmark`, `storage.atomic_json` | Bounded small local state; original source/window validation |

Backend references are under `src/geospatial_search/`; frontend references are
under `frontend/service/`. Bundle GeoParquet metadata declares GeoParquet 1.0.0
and WKB Polygon geometry. Preserve existing versioned formats and legacy readers;
introduce explicit migrations when making incompatible persisted changes.

`status.json` and saved process identifiers are observations. OS locks establish
ownership. A `PAUSE` file requests a transition; successful lock-release
acknowledgement establishes that a worker has finished publishing its active
source. Stale status files do not reserve GPU resources.

## HTTP and UI contracts

The HTTP owner is [http_server.py](../frontend/service/http_server.py). TypeScript
consumer shapes live in [types.ts](../frontend/ui/types.ts); authenticated request
construction is in [api.ts](../frontend/ui/api.ts). TypeScript types do not replace
runtime input validation.

| Route | Input/output contract owner |
| --- | --- |
| `GET /api/config`, `/api/status` | `Application.config/status`, `retrieval.status_snapshot`; map inventory, coverage, model/cache/overview state |
| `POST /api/search` | `retrieval.validate_query/validate_search_options`, `Application.start_search`; task ID response |
| `GET /api/search-progress?id=...` | `Application.search_progress`; loading/encoding/searching/metadata/finishing/complete/error states |
| `GET /api/results?id=...`, `/api/history` | `Application.saved/saved_backend/history`; compatible persisted results and history records |
| `POST /api/model/load`, `/api/model/unload` | `Application.start_model/stop_model`; owned resident lifecycle |
| Tile/preview/inspect GET routes | `TileRenderer`, HTTP routing; finite coordinates, source inventory, native window containment |
| Overview/cache/history mutation routes | HTTP confirmation strings + Application/storage validators; bounded owned artifacts only |

Search requests accept query, `top_k`, partial-collection opt-in, backend/effort,
supplement selection, and seen IDs. Defaults and limits are defined by the
validators, not copied into a new schema. The current HTTP body cap is 32 KiB;
the seen-ID count cap of 20,000 does not guarantee such a list fits that transport.
Plan a separate tested transport change if larger batches are needed.

Scores are float32 dot products against original normalized float16 vectors;
they rank semantic chip similarity and are not object-detection confidence.
ANN candidate selection is approximate; exact search remains the baseline.
Missing/stale optional indexes can fall back to exact search in automatic mode.

## Synthetic and diagnostic contracts

Synthetic jobs explicitly mark fixtures and use a distinct seeded-vector model
contract. Their scores exercise wiring, not semantic quality. Never substitute
synthetic vectors into production collections or present them as GeoRSCLIP.

[doctor.py](../scripts/doctor.py) emits `schema_version: 1` and named observations
with `ok`, `warning`, `error`, or `skipped` status. It checks small metadata and
index size/snapshot by default, without full bundle verification or native-index
loading. Its stdlib index signature is tested against the backend; search remains
the authoritative integrity gate. Optional index checksumming and GPU capability
probing are explicit, do not load a model, and do not reserve locks.
