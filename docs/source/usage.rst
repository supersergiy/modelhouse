Usage
=====

Model creation (upload)
^^^^^^^^^^^^^^^^^^^^^^^
    modelhouse create {source_path} {destination_path} {optional: model type}
    modelhouse set_secret_folder {folder}
    modelhouse delete {model_path} 

Model loading 
^^^^^^^^^^^^^
    import modelhouse
    model = modelhouse.load_model(model_path)


