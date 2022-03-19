#!/bin/bash


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
echo " "
currentDate=`date`
echo $currentDate
echo " "
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
WORD_EMB_PATH='models/word_emb/glove.840B.300d.txt'
EVAL_TYPE='similarity,ranking'
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
echo " "
echo "Word Embedding Path: " $WORD_EMB_PATH
echo "Evaluation Type: " $EVAL_TYPE
echo " "
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_word_evaluation.py       \
        --word_emb_model                $WORD_EMB_PATH \
        --dist_metric                   cos \
        --eval_type                     $EVAL_TYPE \
        --pos_pairs_type                ws \
        --ws_pos_ratio                  0.25 \
        --background_vocab_type         ws

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
