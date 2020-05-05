import os

from modelhouse.loading import load_model
from modelhouse.utils import toabs

MODELHOUSE_DIR = toabs('~/.modelhouse')
MODELHOUSE_TMP_FILES_DIR = os.path.join(MODELHOUSE_DIR, 'tmp_files')
os.makedirs(MODELHOUSE_TMP_FILES_DIR, exist_ok=True)

