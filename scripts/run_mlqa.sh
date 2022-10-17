#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


REPO=$PWD
MODEL=${1:-xlm-roberta-base}
TGT=${2:-mlqa}
GPU=${3:-3}
MODEL_PATH=${4}
SEED=${5:-1}
METHOD=${6:-OIL}
DATA_DIR=${7:-"$REPO/download/"}
OUT_DIR=${8:-"$REPO/outputs/"}


# hyper-para for training

if [ $METHOD == "OIL" ]; then
  memory_size=3 
  topk=100    # --> denoising
  m=0.99    #--> for expert update
  debias=1    # --> do debias or not
  alpha=1    #1 --> control the indirect effect
elif [ $METHOD == "PL" ]; then
  memory_size=3 
  topk=100    #--> denoising
  m=0    #--> for expert update
  debias=0    # --> do debias or not
  alpha=1    # --> control the indirect effect
else
  memory_size=3 
  topk=-1    # --> denoising
  m=0    #--> for expert update
  debias=0    # --> do debias or not
  alpha=1    # --> control the indirect effect
fi

seed=$SEED

LR=1e-6 
BATCH_SIZE=16

MAXL=384
NUM_EPOCHS=100
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
fi

DIR=${DATA_DIR}/${TGT}/
PREDICTIONS_DIR=${MODEL_PATH}/MD2.memory_size=$memory_size.topk=$topk.m=$m.alpha=$alpha.BATCH_SIZE=$BATCH_SIZE.LR=$LR.seed=$seed/predictions
PRED_DIR=${PREDICTIONS_DIR}/$TGT/
mkdir -p "${PRED_DIR}"

if [ $TGT == 'noiseQA-syn'  ]; then
  langs=( asr keyboard translation )
elif [ $TGT == 'noiseQA-na' ]; then
  langs=( asr keyboard translation )
elif [ $TGT == 'noiseQA' ]; then
  langs=( na.asr na.keyboard na.translation  syn.asr syn.keyboard syn.translation )
elif [ $TGT == 'mrqa'  ]; then
  langs=( HotpotQA-dev-from-MRQA  NaturalQuestionsShort-dev-from-MRQA  NewsQA-dev-from-MRQA  SearchQA-dev-from-MRQA TriviaQA-web-dev-from-MRQA )
elif [ $TGT == 'xquad' ]; then
  langs=( en es de el ru tr ar vi th zh hi )
elif [ $TGT == 'mlqa' ]; then
  langs=( en es de ar hi vi zh )
fi

echo "************************"
echo ${MODEL}
echo "************************"

echo
echo "Predictions on $TGT"
for lang in ${langs[@]}; do
  echo "  $lang "
  if [ $TGT == 'noiseQA-syn' ]; then
    TEST_FILE=${DIR}/noiseQA-syn.$lang.json
  elif [ $TGT == 'noiseQA-na' ]; then
    TEST_FILE=${DIR}/noiseQA-na.$lang.json
  elif [ $TGT == 'noiseQA' ]; then
    TEST_FILE=${DIR}/noiseQA.$lang.json
   elif [ $TGT == 'mrqa' ]; then
    TEST_FILE=${DIR}/$lang.json
  elif [ $TGT == 'xquad' ]; then
    TEST_FILE=${DIR}/xquad.$lang.json
  elif [ $TGT == 'mlqa' ]; then
    TEST_FILE=${DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
  fi

  if [ $debias == 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_qa_tta.py \
      --model_type ${MODEL_TYPE} \
      --model_name_or_path ${MODEL_PATH} \
      --do_OIL \
      --debias \
      --memory_size $memory_size \
      --topk $topk \
      --m $m \
      --alpha $alpha \
      --seed $seed \
      --learning_rate $LR \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --num_train_epochs $NUM_EPOCHS \
      --logging_steps 2 \
      --eval_lang ${lang} \
      --predict_file "${TEST_FILE}" \
      --output_dir "${PRED_DIR}" #&> /dev/null
  elif [ $debias == 0 ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_qa_tta.py \
      --model_type ${MODEL_TYPE} \
      --model_name_or_path ${MODEL_PATH} \
      --do_OIL \
      --memory_size $memory_size \
      --topk $topk \
      --m $m \
      --alpha $alpha \
      --seed $seed \
      --learning_rate $LR \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --num_train_epochs $NUM_EPOCHS \
      --logging_steps 2 \
      --eval_lang ${lang} \
      --predict_file "${TEST_FILE}" \
      --output_dir "${PRED_DIR}" #&> /dev/null
  fi

  PRED_FILE=${PRED_DIR}/predictions_${lang}_.json
  if [ $TGT == 'mlqa' ]; then
    python "${PWD}/third_party/evaluate_mlqa.py" "${TEST_FILE}" "${PRED_FILE}" "${lang}"
  fi
done

echo "PRED_FILE: ${PREDICTIONS_DIR}"
echo "Predictions on $TGT"
EVAL_SQUAD=${PWD}/third_party/evaluate_squad.py
if [ $TGT == 'mlqa'  ]; then
  EVAL_SQUAD=${PWD}/third_party/evaluate_mlqa.py
fi
for lang in ${langs[@]}; do
  echo -n "  $lang "
  if [ $TGT == 'noiseQA-syn' ]; then
    TEST_FILE=${DIR}/noiseQA-syn.$lang.json
  elif [ $TGT == 'noiseQA-na' ]; then
    TEST_FILE=${DIR}/noiseQA-na.$lang.json
  elif [ $TGT == 'noiseQA' ]; then
    TEST_FILE=${DIR}/noiseQA.$lang.json
  elif [ $TGT == 'mrqa' ]; then
    TEST_FILE=${DIR}/$lang.json
  elif [ $TGT == 'xquad' ]; then
    TEST_FILE=${DIR}/xquad.$lang.json
  elif [ $TGT == 'mlqa' ]; then
    TEST_FILE=${DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
  fi
  PRED_FILE=${PRED_DIR}/predictions_${lang}_.json

  if [ $TGT == 'mlqa' ]; then
    python "${EVAL_SQUAD}" "${TEST_FILE}" "${PRED_FILE}" "${lang}"
  else
    python "${EVAL_SQUAD}" "${TEST_FILE}" "${PRED_FILE}"
  fi
done

echo "debias: $debias"
echo "memory_size: $memory_size"
echo "topk(gamma): $topk"
echo "m(alpha): $m"
echo "alpha(beta): $alpha"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "seed: $seed"
