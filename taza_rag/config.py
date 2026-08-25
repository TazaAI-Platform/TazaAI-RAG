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
    # Deliberately not chat_model: a model scoring its own output measures self-agreement.
    # gpt-4o-mini judging itself passed answers containing claims absent from the sources.
    # Empty falls back to chat_model.
    judge_model: str = "gpt-5"
    answer_max_chunks: int = 16
    # Evidence budget for generation. Both retrieval paths get the same token cost,
    # so passage retrieval spends its saving on more sources rather than less evidence.
    answer_context_tokens: int = 3000
    # Corrective passes allowed per answer. One pass left roughly a third of flagged
    # claims unresolved, and each round costs a repair call plus a re-verify call.
    verify_max_rounds: int = 3
    # Extract cited facts, then write from that list. One-shot generation left supported
    # facts unused; asking the writer to cover more at once cost the Accuracy gate.
    answer_extract_facts: bool = True

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
