python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
ORIGINAL_EXPERIMENT_NAME=$1

if [[ $ORIGINAL_EXPERIMENT_NAME == *t0_adapt* ]]
then
  CHECKPOINT_STEP=70536 # 65536 (pretrain) + 5000 (t0 adapt)
else
  CHECKPOINT_STEP=65536
fi

EXPERIMENT_NAME=$ORIGINAL_EXPERIMENT_NAME"_t0_eval_"$CHECKPOINT_STEP
CHECKPOINT_DIR="gs://bigscience-t5x/arch_objective_exps_v2/$ORIGINAL_EXPERIMENT_NAME/checkpoint_$CHECKPOINT_STEP"

# directory where the T5X repo is cloned.
T5X_DIR="/home/teven/code/t5x"
export PYTHONPATH=${T5X_DIR}/bigscience/gins

# Logs
LOGS_PATH="/home/teven/logs"
mkdir -p $LOGS_PATH

if [[ $ORIGINAL_EXPERIMENT_NAME == c_dec* ]]
then
  MODEL_GIN_FILE=c_dec_xxl.gin
fi
if [[ $ORIGINAL_EXPERIMENT_NAME == nc_dec* ]]
then
  MODEL_GIN_FILE=nc_dec_xxl.gin
fi
if [[ $ORIGINAL_EXPERIMENT_NAME == enc_dec* ]]
then
  MODEL_GIN_FILE=enc_dec_xxl.gin
fi
if [[ $MODEL_GIN_FILE == "" ]]
then
  echo "Incorrect experiment name $ORIGINAL_EXPERIMENT_NAME, does not start with c_dec/nc_dec/enc_dec"
  exit
fi

MODEL_GIN_FILE=bigscience/gins/$MODEL_GIN_FILE
# EVAL_OUTPUT_DIR=/home/teven/arch_objective_exps_v2/$EXPERIMENT_NAME
EVAL_OUTPUT_DIR="gs://bigscience-t5x/arch_objective_exps_v2/t0_eval/$EXPERIMENT_NAME"
mkdir -p $(dirname $EVAL_OUTPUT_DIR)

# We use offline as loading seqio can be quite long.
if [[ $ORIGINAL_EXPERIMENT_NAME == enc_dec* ]]
then
  HF_DATASETS_OFFLINE=0 python3 ${T5X_DIR}/t5x/eval.py \
    --gin_file="$MODEL_GIN_FILE" \
    --gin_file="bigscience/gins/eval_t0.gin" \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.CHECKPOINT_PATH="'$CHECKPOINT_DIR'" \
    --gin.EVAL_OUTPUT_DIR="'$EVAL_OUTPUT_DIR'" \
    2>&1 | tee $LOGS_PATH/t0_eval_$EXPERIMENT_NAME.txt
else
  HF_DATASETS_OFFLINE=0 python3 ${T5X_DIR}/t5x/eval.py \
    --gin_file="$MODEL_GIN_FILE" \
    --gin_file="bigscience/gins/eval_t0.gin" \
    --gin.CHECKPOINT_PATH="'$CHECKPOINT_DIR'" \
    --gin.EVAL_OUTPUT_DIR="'$EVAL_OUTPUT_DIR'" \
    2>&1 | tee $LOGS_PATH/t0_eval_$EXPERIMENT_NAME.txt
fi