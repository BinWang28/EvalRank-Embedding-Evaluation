#!/bin/bash
#SBATCH --output=./scripts/w_model_3.log
#SBATCH --ntasks=1
#SBATCH --partition=cpu2
#SBATCH -w ttnusa1


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

WORD_EMB_PATH='/data07/binwang/research/EvalRank/models/glove/glove.6B.100d.txt'
EVAL_TYPE='ranking,similarity'
DIST_METRIC='l2'
BG_VOCAB='basic,wiki'
POST_PROCESS='True'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_word_evaluation.py       \
        --word_emb_model                $WORD_EMB_PATH \
        --dist_metric                   $DIST_METRIC \
        --eval_type                     $EVAL_TYPE \
        --background_vocab_type         $BG_VOCAB \
        --post_process                  $POST_PROCESS

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =