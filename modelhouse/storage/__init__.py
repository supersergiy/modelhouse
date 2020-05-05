'''
Code mostly taken from github.com/seung-lab/cloud-volume by Will Silversmith
'''

"""
Storage is a multithreaded key-value object
management client that supports GET, PUT, DELETE,
and LIST operations.

It can support any key-value storage system and
currently supports local filesystem, Google Cloud Storage,
and Amazon S3 interfaces.

Single threaded, Python (preemptive) threads and
green (cooperative) threads are available as

SimpleStorage, ThreadedStorage, and GreenStorage respectively.

Storage is an alias for ThreadedStorage
"""

from .storagemanager import (
    SimpleStorage, ThreadedStorage, GreenStorage,
    DEFAULT_THREADS
)
from .interfaces import reset_connection_pools

# For backwards compatibility
Storage = ThreadedStorage
from .secretmanager import secret_manager
