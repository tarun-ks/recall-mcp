# Recall — semantic shell history MCP

Recall is a locally-running MCP (Model Context Protocol) server that gives
Claude Desktop and Claude Code semantic, context-aware access to your shell
history.

**Status:** pre-alpha. Under active development. Not yet on PyPI.

Planned features:

- Semantic search over zsh, bash, fish, and atuin history
- Project-scoped retrieval, sequence patterns, failure analysis
- First-class secret scrubbing — your secrets stay local and never reach the LLM
- Single-file SQLite + `sqlite-vec` store, ~120 MB embedding model, fully offline
- Published `nl2bash` recall@k benchmark numbers

Install instructions, demo GIF, and benchmark numbers will land before the first
tagged release.

License: MIT.
