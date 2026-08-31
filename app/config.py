"""Central configuration.

Everything tunable lives here and is read from environment variables / a local .env
file. Nothing else in the codebase should call `os.environ` directly - they import
`settings` from this module. Phase 7 introduces a separate `ExperimentConfig` for
values that change *per experiment run*; the values here are process-level defaults.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated vars in the environment
    )

    # --- Generation (Anthropic) ---
    anthropic_api_key: Optional[str] = None
    generation_model: str = "claude-opus-5"
    generation_max_tokens: int = 1024

    # --- Embeddings (local sentence-transformers) ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval ---
    top_k: int = 4

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Data locations ---
    documents_dir: str = "data/documents"
    eval_dataset_path: str = "data/eval_dataset.json"
    experiments_dir: str = "data/experiments"
    experiments_db: str = "data/experiments.db"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed once per process."""
    return Settings()


settings = get_settings()
