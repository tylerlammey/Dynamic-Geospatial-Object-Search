# New York State NAIP acquisition

Download all 3,805 New York 2022 NAIP tiles from NOAA Digital Coast and retain
RGB at the native **0.6 meters/pixel (1.667 pixels/meter)**. The downloader writes
JPEG-compressed GeoTIFFs with an internal lossless validity mask. Bands 1/2/3
are RGB; NIR is omitted. JPEG quality starts at 83 and may fall to 80, 77, then
75 to meet the per-tile size ceiling. Compression does not change pixel spacing.

Approximately **1.6 TB travels over the network** even
though compressed RGB output is bounded at about **169 GB**. The default job
folder has a **300 decimal GB** cap, including temporary files, and a 5 GB reserve.
This acquisition needs CPU, RAM, disk, and network access; it does not use a GPU.

## Setup and commands

Use Python 3.12 and PowerShell on Windows. From the repository root, create
`.venv` if needed and install the small acquisition dependency set:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-nys-naip.txt

# Recommended for a new collection: metadata preparation, then a three-tile pilot.
.\scripts\download-nys-naip.ps1 -Prepare
.\scripts\download-nys-naip.ps1 -Pilot

# Start/resume and show overall and per-worker progress bars.
.\scripts\download-nys-naip.ps1

# Start/resume in the background, or inspect an existing job.
.\scripts\download-nys-naip.ps1 -Background
.\scripts\download-nys-naip.ps1 -Monitor
.\scripts\download-nys-naip.ps1 -Status
```

An existing project environment already contains these dependencies; do not
recreate it during a running job. The default output is anchored to the checkout:
`data/nys-naip-2022-rgb-060m`, regardless of the shell's working directory.
`-OutputDir` selects a different folder; relative paths are resolved against the
repository root. Use an ignored data directory for any alternative output.

Four tile workers are the default. Set `-Workers 1`, `2`, `3`, or `4` when starting
an inactive job. The operating-system lock rejects duplicate workers, and the
launcher attaches to a running job instead of restarting it. Preparation and
pilot mode require an inactive job.

The monitor updates every two seconds with completed tiles, folder usage, worker
phases, ETA, and estimated finish time in Eastern time. ETA uses recent completion
intervals from the current worker session, weighted by remaining RGB pixels.
It starts after the first wave of completions and varies with throughput.
Closing the monitor or pressing Ctrl+C leaves the background download running.
Keep Windows awake, plugged in, and connected. After a reboot, rerun the launcher.

To pause cleanly, create the job's `STOP` file and wait for a paused status:

```powershell
New-Item data\nys-naip-2022-rgb-060m\STOP -ItemType File -Force
.\scripts\download-nys-naip.ps1 -Status
# Explicit resume removes STOP and verifies completed output hashes first.
.\scripts\download-nys-naip.ps1
```

## Existing downloads from the data-local scripts

The tracked launcher uses the same default directory, lock, manifest, imagery,
and provenance schema as the original `data/.../run.ps1` launcher. An already
running download can keep using those original scripts. The tracked launcher
can monitor it immediately and resume its completed tiles after it exits. No
files need to be moved and completed tiles do not need to be downloaded again.

## Local output structure

```text
data/nys-naip-2022-rgb-060m/
  imagery/utm17/*.tif
  imagery/utm18/*.tif
  imagery/utm19/*.tif
  records/*.tif.json
  metadata/                 # Provider index and three VRT snapshots
  temporary/                # At most four source/staged-output pairs
  runtime/                  # Per-session console/error logs
  manifest.json
  status.json
  events.jsonl
  worker.lock
  pilot.json                # Only after a successful pilot
  completion.json           # Only after all 3,805 tiles are verified
```

All these files are ignored by Git. Code is in
`src/geospatial_search/download_nys_naip.py`, launch/monitor scripts in `scripts/`,
and synthetic offline tests in `tests/test_download_nys_naip.py`.

`manifest.json` records source URLs, dimensions, grid, CRS, byte allowances, and
hashes of provider metadata snapshots. Each tile's `records/` entry records its
output path, source HTTP metadata and SHA-256, output SHA-256, compression quality,
dimensions, transform, native CRS, and WGS84 bounds. Interpret coordinates with
their CRS. The collection preserves NAD83 UTM zones 17/18/19 and overlapping
source buffers; it is not a single deduplicated statewide raster. Nominal uniform
pixel spacing does not imply identical ground distance or positional accuracy.

Downloads and compressed writes pass through shared budget accounting. GDAL
encodes in RAM so it cannot bypass the disk guard. Every output retains the source
grid and every source mask pixel; the saved TIFF is fully decoded and checksummed
before publication. A tile that cannot fit at quality 75 stops the job.
Resume verifies saved output hashes before skipping completed tiles. A corrupted
completed output stops verification for investigation. Source partials have
known names and are cleaned only when worker threads have exited.

Atomic file publication retries temporary Windows sharing violations. A locked
`status.json` may leave the display briefly stale while the download continues
and retries; imagery and provenance publication remain mandatory. The folder
cap guards downloader writes and reserves room for logs. It is not an OS quota
for arbitrary external writes: keep unrelated files out of the job folder.

## Verification and embedding compatibility

Run the offline tests without downloading imagery:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

They exercise concurrent byte accounting, bounded scheduling, resume/corruption
checks, file locking, status publication, ETA, and real compression of small
synthetic four-band rasters including partial blocks and invalid edge pixels.
No real imagery, checkpoints, embedding arrays, or generated reports are fixtures.

The existing Albany embedder discovers JP2 imagery and requires one CRS per run.
Statewide embedding still needs GeoTIFF discovery, multiple-CRS handling, and
streamed or sharded metadata planning. This downloader produces no embeddings.
At the current 512-pixel chip / 256-pixel stride baseline, the tile inventory
implies about 6.55 million chips, including overlapping tile buffers. A 1,024-wide
float16 embedding matrix would occupy about 13.4 GB, plus metadata. Keep this
2022 collection separate from the Albany 2024 embedding bundle.

Provider: [NOAA Digital Coast tile index](https://coastalimagery.blob.core.windows.net/digitalcoast/NY_NAIP_2022_9986/index.html).
Dataset metadata and use conditions: [NOAA InPort](https://www.fisheries.noaa.gov/inport/item/71609).
Attribution: Courtesy USDA-FSA; distributed by NOAA Digital Coast. Imagery and
model licenses apply separately from the repository's MIT code license.
