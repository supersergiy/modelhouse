import tempfile
import shutil
import os
import sys
import importlib

import modelhouse
from modelhouse.storage import get_files_and_contents, put_files_and_contents

def import_file(module_name, file_path):
    prev_cwd = os.getcwd()
    os.chdir(os.path.dirname(file_path))

    sys.path.insert(1, os.path.dirname(file_path))
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    os.chdir(prev_cwd)
    return module

def uncached_load_model(path, **params):
    # create temp dir
    tmp_dir_path = tempfile.mkdtemp(dir=modelhouse.MODELHOUSE_TMP_FILES_DIR)

    files_and_contents = get_files_and_contents(path)
    if len(files_and_contents) == 0:
        raise Exception(f"No model found at {path}")
    put_files_and_contents(tmp_dir_path, files_and_contents)
    creator_path = os.path.join(tmp_dir_path, "create.py")
    creator = import_file("create", creator_path)
    model = creator.create(**params)
    shutil.rmtree(tmp_dir_path)
    return model
