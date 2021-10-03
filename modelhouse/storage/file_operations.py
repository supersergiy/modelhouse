import os

from . import Storage

def get_files_and_contents(path):
    with Storage(path) as store:
        files = store.list_files(flat=False)
        read_files = store.get_files(files)

    files_and_contents = []
    for read_file in read_files:
        file_relpath = read_file['filename']
        if read_file['error'] is not None:
            raise Exception("Error while reading {}: {}".format(
                os.path.join(src_path, file_relpath), read_file['error']))

        files_and_contents.append((file_relpath, read_file['content']))
    return files_and_contents

def put_files_and_contents(path, files_and_contents):
    with Storage(path) as store:
        store.put_files(
                files_and_contents,
                content_type='application/octet-stream',
                compress=None,
                cache_control='no-cache'
        )
