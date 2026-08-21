from src.core.plugins.context import PluginContext


def register(context: PluginContext):
    from rerank import rerank

    context.register_search_reranker("cross-encoder", rerank)
    if context._app:
        from routes import create_router

        context.include_router(create_router(context))
