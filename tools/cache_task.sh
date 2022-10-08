#!/bin/bash

# change variables as desired
MODULE=mixtures.c4.span_corruption
TASK_NAME=c4_eye_span_corruption
CACHE_DIR=gs://t5x-test/seqio_cached_tasks

seqio_cache_tasks \
    --module_import=$MODULE \
    --tasks=${TASK_NAME} \
    --output_cache_dir=$CACHE_DIR
    # --pipeline_options="--runner=DataflowRunner,--project=august-clover-363917,--region=us-central2-b,--job_name=download_c4,--staging_location=gs://t5x-test/binaries,--temp_location=gs://t5x-test/tmp,--num_workers=32,--machine="
