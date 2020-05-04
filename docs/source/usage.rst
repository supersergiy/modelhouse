Usage
=====

Model creation (upload)
-----------------------
Trying to make it look like code::

    modelhouse create {source_path} {destination_path} {optional: model type}
    modelhouse set_secret_folder {folder}
    modelhouse delete {model_path} 

Model loading 
-------------
In order to load the model, simply import ``modelhouse`` and calling ``load_model``::
    import modelhouse
    model = modelhouse.load_model(model_path)

Working with Cache
^^^^^^^^^^^^^^^^^^

Change cache paramenters::
    import modelhouse
    modelhouse.clear_cache()
    params = moelhouse.get_cache_params()
    params.algo = modelhouse.cache_algos.LRU 
    params.max_capacity = 10
    modelhouse.set_cache_params(params)



