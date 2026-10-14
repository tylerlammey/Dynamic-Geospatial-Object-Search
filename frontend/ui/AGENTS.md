# UI agent guide

Follow the root and `../AGENTS.md` first. The UI has four explicit boundaries:

- `types.ts`: service payloads; update the service contract reference when
  changing wire fields. Type assertions do not validate untrusted payloads.
- `api.ts`: fragment/session bootstrap and authenticated JSON requests. Keep
  credentials out of logs and remove the session fragment from the address bar.
- `local-tiles.ts`: four-request tile scheduler and Leaflet retention/cancellation.
  Supply route/report callbacks; do not import application state or DOM controls.
- `main.ts`: application state, results/selection rendering, dialogs, and polling.
  Keep tightly coupled polling changes together; extract additional modules when
  a concrete change benefits from an independent boundary.

Run `frontend/scripts/build.ps1` from the root for UI changes. Tile lifecycle
changes also require the zoom/browser checks described in `../AGENTS.md`; use
synthetic fixtures when production imagery is unavailable. Keep decoded tiles
until Leaflet prunes them; cancel unfinished obsolete zoom requests and revoke
object URLs when unloading. Preserve dataset-qualified result identities.

Avoid whole-file formatting of unrelated UI code. New modules should use readable
statements and named types. Coordinate wire changes with the service editor.
