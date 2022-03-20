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
EVAL_TYPE='ranking,similarity,classification'
DIST_METRIC='cos'
POST_PROCESS='False' # 'True', 'False'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_sent_evaluation.py \
        --sent_emb_model                bow \
        --bow_we_path                   models/glove/glove.6B.50d.txt \
        --model_mode                    ori \
        --model_index                   1 \
        --eval_type                     ranking,similarity,classification \
        --dist_metric                   cos \
        --pos_pairs_type                ss,srs \
        --ss_pos_ratio                  0.25 \
        --srs_pos_ratio                 0.25 \
        --background_sent_type          ss,srs

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =