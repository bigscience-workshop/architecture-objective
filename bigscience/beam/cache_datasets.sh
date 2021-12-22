# Need to install seqio
# gcloud auth application-default login

MODULE_IMPORT=beam.task
TASK_NAME=mt0.oscar
JOB_NAME=mt0oscar # the name must consist of only the characters [-a-z0-9], starting with a letter and ending with a letter or number
BUCKET=gs://bigscience-t5x # Don't know is cache needs to be task specific or not ...
PROJECT=bigscience
REGION=us-central2 # TODO: Check if we can have a generic us region
NUM_WORKERS=1000 # TODO: We might need a log more than this

# TODO: One thing we need to figure out is how does it handle HF datasets cache. If all workers need to download it, it's a big no no.

seqio_cache_tasks \
 --module_import=$MODULE_IMPORT \
 --tasks=${TASK_NAME} \
 --output_cache_dir=${BUCKET}/multilingual_t0/v0.3 \
 --pipeline_options="--runner=DataflowRunner,--project=$PROJECT,--region=$REGION,--job_name=$JOB_NAME,--staging_location=$BUCKET/binaries,--temp_location=$BUCKET/tmp,--setup_file=$PWD/setup.py,--num_workers=$NUM_WORKERS,--autoscaling_algorithm=NONE,--machine_type=n1-highmem-2"