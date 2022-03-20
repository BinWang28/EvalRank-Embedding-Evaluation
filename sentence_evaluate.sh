#!/bin/bash

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

WORD_EMB_PATH='models/word_emb/toy_emb.txt' # 'models/word_emb/glove.840B.300d.txt'
EVAL_TYPE='ranking,similarity' # 'similarity', 'ranking'
DIST_METRIC='cos' # 'cos', 'l2'
BG_VOCAB='basic,wiki' # 'basic', 'wiki'
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