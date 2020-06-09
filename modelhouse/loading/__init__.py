import cachetools
from .loading import uncached_load_model_str

load_model = cachetools.cached(cachetools.LRUCache(maxsize=8))(uncached_load_model_str)
