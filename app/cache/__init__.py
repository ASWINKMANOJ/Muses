# app/cache/__init__.py
from app.cache.query_cache import get_query_cache, SemanticQueryCache

__all__ = ["get_query_cache", "SemanticQueryCache"]
