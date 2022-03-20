#!/bin/bash

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

SENT_EMB_MODEL='bow'
BOW_WE_PATH='src/models/toy_emb.txt'
BOW_WE_PATH='/data07/binwang/research/EvalRank/models/glove/glove.6B.50d.txt'
EVAL_TYPE='ranking,similarity,classification' # ranking,similarity,classification
DIST_METRIC='cos'
POST_PROCESS='False' # 'True', 'False'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_sent_evaluation.py \
        --sent_emb_model                $SENT_EMB_MODEL \
        --bow_we_path                   $BOW_WE_PATH \
        --eval_type                     $EVAL_TYPE \
        --dist_metric                   $DIST_METRIC

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =