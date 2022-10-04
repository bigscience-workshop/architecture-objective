python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
CHECKPOINT_DIR=$1

EXPERIMENT_DIR=$(dirname "$CHECKPOINT_DIR")
EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
# directory where the T5X repo is cloned.
T5X_DIR="~/code/architecture-objective"
export PYTHONPATH=${T5X_DIR}/bigscience/gins

# Logs
LOGS_PATH="~/logs"
mkdir -p $LOGS_PATH

if [[ $EXPERIMENT_NAME == c_dec* ]]
then
  MODEL_GIN_FILE=c_dec_xxl.gin
fi
if [[ $EXPERIMENT_NAME == nc_dec* ]]
then
  MODEL_GIN_FILE=nc_dec_xxl.gin
fi
if [[ $EXPERIMENT_NAME == enc_dec* ]]
then
  MODEL_GIN_FILE=enc_dec_xxl.gin
fi
if [[ $MODEL_GIN_FILE == "" ]]
then
  echo "Incorrect experiment name $EXPERIMENT_NAME, does not start with c_dec/nc_dec/enc_dec"
  exit
fi

MODEL_GIN_FILE=bigscience/gins/$MODEL_GIN_FILE
echo "Running the following config: $MODEL_GIN_FILE" 2>&1 | tee $LOGS_PATH/pretrain_$EXPERIMENT_NAME.txt
LOCAL_MODEL_DIR=~/model_dir

python3 ${T5X_DIR}/bigscience/scripts/inference_tool.py \
  --gin_file="$MODEL_GIN_FILE" \
  --gin_file="bigscience/gins/inference_tool.gin" \
  --gin.MODEL_DIR="'${LOCAL_MODEL_DIR}'" \
  --gin.INITIAL_CHECKPOINT_PATH="'${CHECKPOINT_DIR}'" \
    2>&1 | tee -a $LOGS_PATH/pretrain_$EXPERIMENT_NAME.txt

# sh bigscience/scripts/infer.sh {CHECKPOINT_DIR}