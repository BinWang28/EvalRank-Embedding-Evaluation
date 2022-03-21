#!/bin/bash
#SBATCH --output=./scripts/sentence_l2/s_14.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -w ttnusa2

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = simcse

SENT_EMB_MODEL='simcse'
MODEL_SPEC='princeton-nlp/unsup-simcse-bert-base-uncased'
EVAL_TYPE='ranking,similarity,classification' # ranking,similarity,classification
DIST_METRIC='l2' # only for ranking

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_sent_evaluation.py \
        --sent_emb_model                $SENT_EMB_MODEL \
        --model_spec                    $MODEL_SPEC \
        --eval_type                     $EVAL_TYPE \
        --dist_metric                   $DIST_METRIC

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =