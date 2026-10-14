# Developer and operating scripts

Root AGENTS.md applies. PowerShell entry points resolve repository paths from
`$PSScriptRoot`, propagate native process failures, and restore environment
changes in `finally`. Quote paths and support spaces and calls from another
working directory. Use native PowerShell filesystem operations and verify
absolute containment before recursive removal; never delete computed roots.

Keep developer tooling independent of production jobs. `setup-dev.ps1` installs
only into `.venv-dev`; validation does not install anything. Diagnostics use
stdlib metadata and read-only probes by default. Synthetic output belongs only
under ignored `temp/` or `frontend/temp/`, with no overwrite or redirected paths.
CPU tensor tests do not authorize CPU production inference. `GpuSmoke` must use
a new bounded smoke collection and query that exact collection, never a full run.

See [development workflow](../docs/DEVELOPMENT.md) for commands. Add tests for
failure propagation, containment, skipped critical coverage, and diagnostics
that could otherwise affect user state. Stop only processes created by a check.
