# Service agent guide

Follow the root and `../AGENTS.md` first. Use these boundaries to find the change:

- `server.py`: `Application` orchestration, owned workers, overview, saved state;
  executable entry point and ownership context. `Server`/`Handler` are re-exported
  for compatibility with existing callers.
- `http_server.py`: HTTP transport, authenticated loopback/origin checks, bounded
  connections, response security headers, static containment, route dispatch.
  It accepts an application without importing or constructing one.
- `imagery.py`, `overview.py`: read-only sources and bounded display rendering.
- `storage.py`: containment, atomic state writes, bounded cache, OS ownership.
- `retrieval.py` and worker modules: published collection/model integration.

Run the frontend offline suite with `PYTHONPATH=src;frontend/service`. Test HTTP
boundaries with isolated fake applications and ephemeral loopback ports, as in
`tests/test_http_server.py`; these need no real imagery, weights, or GPU. Preserve
authorization before dispatch, static path containment, and cache/process owners.

Changing request payloads requires meaningful validation tests plus updates to
UI types and the service contract reference. Never start/stop a borrowed service
in tests or modify a user's model worker. Keep frozen backend recipes intact.
