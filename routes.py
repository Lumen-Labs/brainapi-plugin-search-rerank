from fastapi import APIRouter

from src.core.plugins.context import PluginContext


def create_router(context: PluginContext) -> APIRouter:
    router = APIRouter(prefix="/search-rerank", tags=["search-rerank-plugin"])

    @router.get("/health")
    def health() -> dict:
        from rerank import status

        return status()

    return router
