#!/bin/bash


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

WORD_EMB_PATH='models/word_emb/glove.840B.300d.txt'
EVAL_TYPE='similarity,ranking'
DIST_METRIC='cos'
BG_VOCAB='basic,wiki'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_word_evaluation.py       \
        --word_emb_model                $WORD_EMB_PATH \
        --dist_metric                   $DIST_METRIC \
        --eval_type                     $EVAL_TYPE \
        --background_vocab_type         $BG_VOCAB

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
