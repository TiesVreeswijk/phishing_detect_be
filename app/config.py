from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()