# Backend synthetic tests

Root AGENTS.md applies. Fixtures are legally shareable synthetic rasters and
vectors, generated in temporary directories; no NYS pixels or checkpoints.
CPU PyTorch may exercise tensor plumbing without a real model. Never bypass
production GPU checks, compatibility validation, locks, or budget guards just
to make a test pass. Use meaningful behavioral assertions rather than copies
of implementation expressions.

`scripts/validate.ps1 -Mode Offline` runs this suite and frontend tests. Its
runner requires Faiss/threadpoolctl and rejects index-test skips. Explicit
individual unittest commands remain useful for focused diagnosis; report their
coverage and skips accurately. Windows ownership/sharing semantics are tested
on Windows CI. Keep generated fixtures, screenshots, profiles, and reports out
of Git. Prefer extending existing fixture helpers over divergent contracts.
