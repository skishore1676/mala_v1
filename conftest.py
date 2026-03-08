"""Root conftest.py — tells pytest to skip problematic paths during collection."""

collect_ignore_glob = [".env", ".env.*", "data/*", ".data_cache/*"]
