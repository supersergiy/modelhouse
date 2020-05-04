Usage
=====

Model creation (upload)
^^^^^^^^^^^^^^^^^^^^^^^
Trying to make it look like code::

    modelhouse create {source_path} {destination_path} {optional: model type}
    modelhouse set_secret_folder {folder}
    modelhouse delete {model_path} 

Model loading 
^^^^^^^^^^^^^
In order to load the model, simply import ``modelhouse`` and calling ``load_model``::
    import modelhouse
    model = modelhouse.load_model(model_path)


