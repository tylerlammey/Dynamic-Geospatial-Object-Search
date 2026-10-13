# NYS imagery gap supplement

The original 3,805-tile NYS 2022 inventory and embedding collection remain
unchanged. A separate 75-tile supplement fills the two audited interior land
gaps using NOAA's 2021 release. Eastern tiles were acquired in
September/November 2021; western tiles were acquired on April 29, 2022. Release
year and acquisition date are distinct provenance fields. This is mixed-date
imagery, not a new strictly 2022 acquisition.

Sources: [NOAA download index](https://coastalimagery.blob.core.windows.net/digitalcoast/NY_NAIP_2021_9664/index.html),
[dataset metadata](https://www.fisheries.noaa.gov/inport/item/69001).
Attribution: Courtesy USDA-FSA; distributed by NOAA Digital Coast.

## Commands

Preparation requires the local, verified gap audit and preserved provider index
at `data/nys-naip-2022-rgb-060m/verification/gap-patch-audit-20261012/`
(`verification.json` and `noaa-2021-index.html`). These provenance files are
ignored by Git. A fresh computer needs that audit transferred with the job,
or a new audit, before preparing the supplement. Existing manifests support
download resume without preparing again; this downloader does not discover
new gaps automatically.

Run from the repository root:

```powershell
.\scripts\download-nys-gapfill.ps1 -Prepare
.\scripts\download-nys-gapfill.ps1 -Pilot -Background
.\scripts\download-nys-gapfill.ps1 -Monitor
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m geospatial_search.verify_gapfill --pilot

# After inspecting the pilot:
.\scripts\download-nys-gapfill.ps1 -Background
.\scripts\download-nys-gapfill.ps1 -Monitor
.\.venv\Scripts\python.exe -B -m geospatial_search.verify_gapfill
.\scripts\embed-nys-gapfill.ps1 -Plan
.\scripts\embed-nys-gapfill.ps1 -SmokeTest -Offline
# Test the printed smoke path using search-nys.ps1 -CollectionDir PATH,
# and -ImageryDir data/nys-naip-2022-rgb-060m/supplements/naip-2021-release-gapfill/imagery.
.\scripts\embed-nys-gapfill.ps1 -Background -Offline
.\scripts\embed-nys-gapfill.ps1 -Monitor
.\scripts\embed-nys-gapfill.ps1 -Pause
.\scripts\search-nys.ps1 -Query 'forest clearing' -Offline
.\scripts\embed-nys-gapfill.ps1 -Resume -Background -Offline
```

`-Status` prints one update. Closing a monitor leaves its worker running.
Download resume validates completed file checksums and restarts incomplete
source files. Touch `STOP` in the supplement folder to stop its download;
the next launch clears that signal. Embedding pause finishes and publishes
the active TIFF before releasing the shared GPU lock. Repeat a background
embedding launch to attach to a live worker. The launcher keeps Windows awake
for the active job without changing persistent power settings. Unload a
resident frontend model before embedding or CLI queries.

## Data and compatibility

All new acquisition/embedding artifacts stay under:

```text
data/nys-naip-2022-rgb-060m/supplements/naip-2021-release-gapfill/
  manifest.json           # Audited subset, URLs, dates, grids, target holes
  completion.json         # Manifest hash and verified completion count
  metadata/index.html
  imagery/utm18/*.tif     # Native JPEG-compressed RGB and original masks
  records/*.tif.json      # HTTP metadata, hashes, transforms, provenance
  status.json
  runtime/*.log
  verification/           # Full-mask coverage and pilot native previews
  tiling/native060cm-s256-st192/plan.json
  embeddings/georsclip-vith14-s256-st192/
    run.json              # Independent recipe/fingerprint
    collection.json       # Atomically published source shards
    status.json           # Percent, elapsed runtime, ETA, saved vectors
    worker.lock
    runtime/*.log
    shards/SOURCE_TILE/    # Matching NumPy/GeoParquet/run bundle
  embeddings/smoke-TIMESTAMP/
```

The primary job's one 300 GB Budget covers all supplement output and scratch.
There is no second quota. The separate 2021 transfer allowlist leaves the
original downloader's 2022 guards intact. Compression uses the existing
83/80/77/75 quality policy and validates every mask pixel. Native 0.6 m spacing,
RGB bands, and NAD83 UTM18 transforms are preserved.

The measured source download is 31.22 GB; compressed RGB TIFFs total 1.94 GB.
Full decoded-mask verification measured 100% coverage of both target holes.
This is not a full audit of other coastal/state-boundary omissions or cloud
quality throughout New York.

The same pinned GeoRSCLIP weights, preprocessing, float32 normalization,
float16 storage, 256-pixel chip, 192-pixel stride, and 95% source-validity rule
apply. Chip centers must fall inside the target gaps buffered by 76.8 meters,
preserving native boundary context while limiting redundant older buffers.
Geographic eligibility is separate from mask validity. No source mask is
modified artificially, and accepted windows remain full size.

The supplement adapter reuses the durable engine through bindings scoped to
its dedicated process. Original embedding/model/tiling/bundle files and their
stored hashes stay unchanged. The supplement recipe also fingerprints its
adapter and inventory helpers; do not edit them during a running/resumable job.

## Map and search

Gap-fill pixels are composited underneath valid base imagery. Both manifests
contribute to the combined display fingerprint, while original per-source
photographic caches are reused. The old base statewide overview remains visible
while the combined overview is prepared automatically when embedding is idle.
Display caches remain bounded under `frontend/cache/`.

Default frontend/CLI searches encode text once, verify each collection's model
contract, retrieve top matches, and merge original-vector cosine scores. The
existing fast index serves the unchanged base collection. The small supplement
uses bounded exact ranking. IDs are globally qualified by
`(dataset_id, shard_id, embedding_id)`; legacy base exclusion pairs still work.
Results preserve per-collection fingerprints and original source URIs.
Historical searches and bookmarks remain readable.

For strict acquisition tests, select **Settings > Use only the original 2022
collection** to exclude supplements from imagery and new searches, or run:

```powershell
.\scripts\search-nys.ps1 -Query 'fighter jet' -BaseOnly -Offline
```

No base index rebuild or re-embedding is needed. Imagery, embeddings, caches,
reports, and previews remain ignored by Git.

## Provenance and change history

The pre-patch baseline commit is `bda1c188e1c2ff36e56b6a99d1c7e9a5c8fd5938`.
`data/nys-naip-2022-rgb-060m/verification/gapfill-rollback/` preserves before/after
snapshots for every patch-owned code file, protected base metadata hashes,
all original imagery checksum records, original embedding bundle hashes,
created search-result folders, and new display-cache namespaces. Generated
build/test artifacts are listed separately: their earlier versions were not
backed up before tests, but can be regenerated from the restored source.

The temporary rollback tooling was removed after the patch was validated and
accepted. The ignored snapshots remain as historical evidence, rather than an
active restoration mechanism. Published code changes are tracked by Git;
imagery, metadata, verification reports, and embedding bundles retain their
separate local provenance.
