from modelhouse.storage import Storage

def uncached_load_model(model_path):
    store = Storage(model_path)
    files = store.list_files(flat=False)
    import pdb; pdb.set_trace()
