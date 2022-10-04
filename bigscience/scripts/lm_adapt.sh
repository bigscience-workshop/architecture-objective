python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
ORIGINAL_EXPERIMENT_NAME=$1
CHECKPOINT_STEP=$2
# The idea is whether to load the nc_dec as c_dec or vice-versa. "true" will switch
OPPOSITE_ARCHITECTURE=$3

# directory where the T5X repo is cloned.
T5X_DIR="~/code/architecture-objective"
export PYTHONPATH=${T5X_DIR}/bigscience/gins

# Logs
LOGS_PATH="~/logs"
mkdir -p $LOGS_PATH

if [[ $ORIGINAL_EXPERIMENT_NAME != *span_corruption* ]]
then
  echo "Incorrect original experiment name, lm adapt should only be used on \`span_corruption\` checkpoints, got $ORIGINAL_EXPERIMENT_NAME"
  exit
fi

EXPERIMENT_NAME=$ORIGINAL_EXPERIMENT_NAME"_lm_adapt_"$CHECKPOINT_STEP
if [[ $ORIGINAL_EXPERIMENT_NAME == c_dec* ]]
then
  if [[ $OPPOSITE_ARCHITECTURE != true ]]
  then
    GIN_FILE=c_dec_c4_lm_adapt.gin
  else
    echo "Using opposite architecture"
    GIN_FILE=nc_dec_c4_lm_adapt.gin
    EXPERIMENT_NAME=$ORIGINAL_EXPERIMENT_NAME"_plm_adapt_"$CHECKPOINT_STEP
  fi
fi
if [[ $ORIGINAL_EXPERIMENT_NAME == nc_dec* ]]
then
  if [[ $OPPOSITE_ARCHITECTURE != true ]]
  then
    GIN_FILE=nc_dec_c4_lm_adapt.gin
  else
    echo "Using opposite architecture"
    GIN_FILE=c_dec_c4_lm_adapt.gin
    EXPERIMENT_NAME=$ORIGINAL_EXPERIMENT_NAME"_flm_adapt_"$CHECKPOINT_STEP
  fi
fi
if [[ $ORIGINAL_EXPERIMENT_NAME == enc_dec* ]]
then
  if [[ $OPPOSITE_ARCHITECTURE != true ]]
  then
    GIN_FILE=enc_dec_c4_lm_adapt.gin
  else
    echo "Cannot have opposite architecture for enc dec."
    exit
  fi
fi
if [[ $GIN_FILE == "" ]]
then
  echo "Incorrect experiment name $ORIGINAL_EXPERIMENT_NAME, does not start with c_dec/nc_dec/enc_dec"
  exit
fi

CHECKPOINT_DIR="gs://bigscience-t5x/arch_objective_exps_v2/$ORIGINAL_EXPERIMENT_NAME/checkpoint_$CHECKPOINT_STEP"
MODEL_DIR="gs://bigscience-t5x/arch_objective_exps_v2/$EXPERIMENT_NAME"

GIN_FILE=bigscience/gins/$GIN_FILE
echo "Running the following config: $GIN_FILE" 2>&1 | tee $LOGS_PATH/pretrain_$EXPERIMENT_NAME.txt

python3 ${T5X_DIR}/t5x/train.py \
  --gin.INITIAL_CHECKPOINT_PATH="'${CHECKPOINT_DIR}'" \
  --gin_file="$GIN_FILE" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  2>&1 | tee -a $LOGS_PATH/pretrain_$EXPERIMENT_NAME.txt

# bash bigscience/scripts/lm_adapt.sh c_dec_c4_span_corruption_bs_128 420000
