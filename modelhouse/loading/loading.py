import tempfile
import shutil
import os
import importlib

import modelhouse
from modelhouse.storage import get_files_and_contents, put_files_and_contents

def import_file(module_name, file_path):
    prev_cwd = os.getcwd()
    os.chdir(os.path.dirname(file_path))

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    os.chdir(prev_cwd)
    return module

def uncached_load_model(model_path, model_params={}):
    # create temp dir
    tmp_dir_path = tempfile.mkdtemp(dir=modelhouse.MODELHOUSE_TMP_FILES_DIR)

    files_and_contents = get_files_and_contents(model_path)
    put_files_and_contents(tmp_dir_path, files_and_contents)
    creator_path = os.path.join(tmp_dir_path, "create.py")
    creator = import_file("create", creator_path)
    model = creator.create(**model_params)
    shutil.rmtree(tmp_dir_path)
    return model
