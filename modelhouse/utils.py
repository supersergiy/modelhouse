import os
import time

def toabs(path):
    path = os.path.expanduser(path)
    return os.path.abspath(path)

