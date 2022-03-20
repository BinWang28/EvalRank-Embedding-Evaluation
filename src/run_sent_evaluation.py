#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 15:27:52
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import sys
import logging
import argparse

import s_data_loader
import s_models
import s_evaluation

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    ''' this is the main file for sentence embedding evaluation '''

    # - - - - - - - - - - - - - - - - -
    # arguments
    parser = argparse.ArgumentParser()
    # Base
    parser.add_argument("--sent_emb_model", type=str, default=None,
                        help="name of sentence embedding mdoel, (bow, infersent, bert, bert-flow, simcse, sbert)")
    parser.add_argument("--eval_type", type=str, default=None,
                        help="evaluation types: similarity,ranking,classification")   
    parser.add_argument("--dist_metric", type=str, default='cos',
                        help="distance measure metric for the ranking evaluation only: cos, l2")  
    # bow parameters
    parser.add_argument("--bow_we_path", type=str, default=None,
                        help="word embedding path for bow model")
    # infersent parameters
    parser.add_argument("--infersent_version", type=str, default='v1',
                        help="version for infersent model")
    # model specification
    parser.add_argument("--model_spec", type=str, default=None,
                        help="model specifications")
    # bert only
    parser.add_argument("--pooler", type=str, default=None,
                        help="pooling method for bert models (cls, last-avg, first-last-avg)")

    config = parser.parse_args()

    # - - - - - - - - - - - - - - - - -
    if config.sent_emb_model == 'bow' and config.bow_we_path == None:
        sys.exit("Error: Must specify the word embedding path if using BOW model")
    if config.eval_type == None:
        sys.exit("Error: Must specify at least one evaluation task")

    config.eval_type  = config.eval_type.split(',')

    # - - - - - - - - - - - - - - - - -
    # selection normalization mode
    if config.dist_metric == 'cos':
        config.normalization = True
    elif config.dist_metric == 'l2':
        config.normalization = False
    else:
        sys.exit("Distance Metric NOT SUPPORTED for ranking: {}".format(config.dist_metric))

    # - - - - - - - - - - - - - - - - -
    # display parameters
    logging.info("*** Parameters ***")
    for item, value in vars(config).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")

    # - - - - - - - - - - - - - - - - -
    # load data
    if 'ranking' in config.eval_type:
        sent_pairs_data = s_data_loader.Sent_ranking_dataset_loader(config)
    else:
        sent_pairs_data = None

    # load model
    sent_emb_model = s_models.Sent_embedding_model(config)
    
    # eval
    our_evaluator = s_evaluation.Sent_emb_evaluator(config, sent_pairs_data, sent_emb_model)
    our_evaluator.eval()

