"""RAG query layer (Phase 5): read-only NL->SQL over the warehouse + optional
semantic recall over weekly reports. The LLM (when one is wired in) only ever
summarizes rows the database actually returned -- it never fabricates."""
