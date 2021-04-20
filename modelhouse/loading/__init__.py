import cachetools
from .loading import uncached_load_model_str, uncached_load_model

load_model = cachetools.cached(cachetools.LRUCache(maxsize=8))(uncached_load_model_str)
load_model_simple = cachetools.cached(cachetools.LRUCache(maxsize=8))(uncached_load_model)
