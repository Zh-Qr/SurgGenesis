# Common Utilities

This folder is reserved for shared Python utilities during migration.

Current refactor strategy uses compatibility wrappers first (calling scripts in `WAN/`).
As logic is migrated from legacy scripts, place reusable code here:
- config loading
- path checks
- logger setup
- dataset discovery
