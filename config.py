"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings(BaseSettings):
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    eleven_api_key: str = os.getenv("ELEVEN_API_KEY", "")
    fish_api_key: str = os.getenv("FISH_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")

    # Integrations / Connectors
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    slack_token: str = os.getenv("SLACK_TOKEN", "")
    linkedin_token: str = os.getenv("LINKEDIN_TOKEN", "")
    apify_token: str = os.getenv("APIFY_TOKEN", "")
    email_token: str = os.getenv("EMAIL_TOKEN", "")

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # ChromaDB
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Upload settings
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    allowed_extensions: str = os.getenv("ALLOWED_EXTENSIONS", ".pdf")

    # LLM Configuration
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    system_prompt: str = os.getenv("SYSTEM_PROMPT", "")

    # Embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    # Upload directory
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")

    # Answering method
    answering_method: str = os.getenv("ANSWERING_METHOD", "tool_calling")

    @property
    def is_groq_configured(self) -> bool:
        return bool(self.groq_api_key and "placeholder" not in self.groq_api_key.lower())

    @property
    def is_google_configured(self) -> bool:
        return bool(self.google_api_key and "placeholder" not in self.google_api_key.lower())

    @property
    def is_tavily_configured(self) -> bool:
        return bool(self.tavily_api_key and "placeholder" not in self.tavily_api_key.lower())

    @property
    def is_mistral_configured(self) -> bool:
        return bool(self.mistral_api_key and "placeholder" not in self.mistral_api_key.lower())

    @property
    def is_llm_configured(self) -> bool:
        if self.llm_provider in ["google", "gemini"]:
            return self.is_google_configured
        if self.llm_provider == "mistral":
            return self.is_mistral_configured
        return self.is_groq_configured

    @property
    def is_demo_mode(self) -> bool:
        if self.llm_provider in ["google", "gemini"]:
            return not (self.is_google_configured and self.is_tavily_configured)
        if self.llm_provider == "mistral":
            return not (self.is_mistral_configured and self.is_tavily_configured)
        return not (self.is_groq_configured and self.is_google_configured and self.is_tavily_configured)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Load persistent settings if they exist
try:
    import json
    from pathlib import Path
    _settings_path = Path("settings.json")
    if _settings_path.exists():
        with open(_settings_path, "r", encoding="utf-8") as _f:
            _persisted = json.load(_f)
            for _k, _v in _persisted.items():
                if hasattr(settings, _k):
                    setattr(settings, _k, _v)
                # Sync back to environment variables for dynamic queries
                _env_key = _k.upper()
                if _v is not None:
                    os.environ[_env_key] = str(_v)
except Exception:
    pass

# Ensure directories exist
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
