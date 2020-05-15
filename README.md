# modelhouse

## Why?

**The purpose of `modelhouse` is to make it easy to distribute models to workers on cloud.**

With `modelhouse`, you can store and load your models from different storage backends (Google Cloud Storage, AWS S3, local). `modelhouse` also provides model caching, which can significantly reduce model loading time while keeping memory utilization low.

## Installation

```pip install modelhouse```

## Documentation
For documentation, please visit https://modelhouse.readthedocs.io/en/latest/

## Getting started
TODO

## TODO:

1) implement model caching through https://cachetools.readthedocs.io/en/stable/#module-cachetools
2) test cloud download/upload
3) finish documentation
4) test different secret folder
