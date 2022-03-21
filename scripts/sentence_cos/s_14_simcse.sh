#!/bin/bash
#SBATCH --output=./scripts/sentence_cos/s_14.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=new
#SBATCH -w hlt06

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = simcse

SENT_EMB_MODEL='simcse'
MODEL_SPEC='princeton-nlp/unsup-simcse-bert-base-uncased'
EVAL_TYPE='ranking,similarity,classification' # ranking,similarity,classification
DIST_METRIC='cos' # only for ranking

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_sent_evaluation.py \
        --sent_emb_model                $SENT_EMB_MODEL \
        --model_spec                    $MODEL_SPEC \
        --eval_type                     $EVAL_TYPE \
        --dist_metric                   $DIST_METRIC

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =