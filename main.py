from __future__ import annotations

import logging

from fastapi import FastAPI

from agent.api.routes import router
from agent.config import get_settings, validate_obsidian_vault_path
from agent.database import initialize_database
from agent.services.ollama import OllamaClient


logger = logging.getLogger("atlas")
logger.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    settings = get_settings()
    validate_obsidian_vault_path(settings.obsidian_vault_path)
    initialize_database(settings.database_path)

    app = FastAPI(title="Atlas", version="0.2.0")
    app.state.settings = settings
    logger.info("Atlas chat model: %s", settings.chat_model)
    logger.info("Atlas embedding model: %s", settings.embedding_model)
    app.state.ollama = OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
    )
    app.include_router(router)
    return app


app = create_app()
