#!/bin/bash

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

WORD_EMB_PATH='models/word_emb/toy_emb.txt' # 'models/word_emb/glove.840B.300d.txt'
EVAL_TYPE='similarity,ranking' # 'similarity', 'ranking'
DIST_METRIC='cos' # 'cos', 'l2'
BG_VOCAB='basic,wiki' # 'basic', 'wiki'
POST_PROCESS='False' # 'True', 'False'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_word_evaluation.py       \
        --word_emb_model                $WORD_EMB_PATH \
        --dist_metric                   $DIST_METRIC \
        --eval_type                     $EVAL_TYPE \
        --background_vocab_type         $BG_VOCAB \
        --post_process                  $POST_PROCESS

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
