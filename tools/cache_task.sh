#!/bin/bash

# change variables as desired
MODULE=mixtures.c4.span_corruption
TASK_NAME=c4_eye_span_corruption
CACHE_DIR=gs://t5x-test/seqio_cached_tasks

seqio_cache_tasks \
    --module_import=$MODULE \
    --tasks=${TASK_NAME} \
    --output_cache_dir=$CACHE_DIR
