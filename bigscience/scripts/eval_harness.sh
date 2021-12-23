pushd ~/code/t5x

ORIGINAL_EXPERIMENT_NAME=$1

if [[ $ORIGINAL_EXPERIMENT_NAME == *t0_adapt* ]]
then
  CHECKPOINT_STEP=37768 # 32768 (pretrain) + 5000 (t0 adapt)
else
  CHECKPOINT_STEP=32768
fi

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
EVAL_OUTPUT_DIR=gs://bigscience-t5x/arch_objective_exps_v2/eai_eval/"$EXPERIMENT_NAME".json

HF_DATASETS_OFFLINE=1 PYTHONPATH=$(pwd)/bigscience/gins python3 $(pwd)/t5x/eval_harness.py \
   --gin_file_="$MODEL_GIN_FILE" \
   --gin_file_="bigscience/gins/eval_harness.gin" \
   --gin.INFER_OUTPUT_DIR="'.'"  \
   --gin.DROPOUT_RATE=0.0 \
   --gin.CHECKPOINT_PATH="'gs://bigscience-t5x/arch_objective_exps_v2/$ORIGINAL_EXPERIMENT_NAME/checkpoint_$CHECKPOINT_STEP'" \
   --results_path $EVAL_OUTPUT_DIR \
   --tasks=arc_challenge,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc
