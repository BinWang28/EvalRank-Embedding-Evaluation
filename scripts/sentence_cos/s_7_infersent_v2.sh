#!/bin/bash
#SBATCH --output=./scripts/sentence_cos/s_7.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -w ttnusa2

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = infersent v2

SENT_EMB_MODEL='infersent'
INFERSENT_VERSION='v2'
INFERSENT_MODEL_PATH='/data07/binwang/research/EvalRank/models/infersent/infersent2.pkl'
INFERSENT_EMB_PATH='/data07/binwang/research/EvalRank/models/fasttext/crawl-300d-2M.vec'
EVAL_TYPE='ranking,similarity,classification' # ranking,similarity,classification
DIST_METRIC='cos' # only for ranking

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_sent_evaluation.py \
        --sent_emb_model                $SENT_EMB_MODEL \
        --infersent_version             $INFERSENT_VERSION \
        --infersent_model_path          $INFERSENT_MODEL_PATH \
        --infersent_emb_path            $INFERSENT_EMB_PATH \
        --eval_type                     $EVAL_TYPE \
        --dist_metric                   $DIST_METRIC

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =