from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    contextualize_model: str = "gpt-4o-mini"
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/index")

    # Retrieval defaults
    chunk_tokens: int = 450
    chunk_overlap_tokens: int = 60
    retrieve_dense_k: int = 40
    retrieve_sparse_k: int = 40
    retrieve_fuse_k: int = 50
    rerank_top_k: int = 10
    answer_max_chunks: int = 8

    # Factiva RAG API
    factiva_rag_client_id: str = ""
    factiva_rag_username: str = ""
    factiva_rag_password: str = ""
    factiva_rag_portal_user: str = ""
    factiva_rag_portal_password: str = ""

    # Factiva News Feed
    factiva_feed_client_id: str = ""
    factiva_feed_username: str = ""
    factiva_feed_password: str = ""
    factiva_feed_portal_user: str = ""
    factiva_feed_portal_password: str = ""

    factiva_token_url: str = "https://accounts.dowjones.com/oauth2/v1/token"
    factiva_api_base: str = "https://api.dowjones.com"
    factiva_device_id: str = "taza-rag-trial"
    factiva_application_id: str = "taza-rag"
    factiva_metrics_user_id: str = Field(
        default="taza_rag_trial_user_md5_placeholder01",
        description="Stable 32-char metrics user id for Factiva usage logging",
    )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
