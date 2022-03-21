#!/bin/bash
#SBATCH --output=./scripts/sentence_l2/s_3.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -w ttnusa2

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

SENT_EMB_MODEL='bow_pp'
BOW_WE_PATH='/data07/binwang/research/EvalRank/models/glove/glove.840B.300d.txt'
EVAL_TYPE='ranking,similarity,classification' # ranking,similarity,classification
DIST_METRIC='l2' # only for ranking

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = bow

python src/run_sent_evaluation.py \
        --sent_emb_model                $SENT_EMB_MODEL \
        --bow_we_path                   $BOW_WE_PATH \
        --eval_type                     $EVAL_TYPE \
        --dist_metric                   $DIST_METRIC

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =