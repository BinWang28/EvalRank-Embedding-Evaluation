#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-18 11:18:44
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

import w_data_loader
import w_model
import w_evaluation

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    # - - - - - - - - - - - - - - - - -
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_emb_model", type=str, default=None,
                        help="name of word embedding model with full path (.txt file supported)")
    parser.add_argument("--dist_metric", type=str, default='cos',
                        help="distance measure between embeddings: cos, l2")
    parser.add_argument("--eval_type", type=str, default=None,
                        help="evaluation types: similarity,ranking")
    parser.add_argument("--background_vocab_type", type=str, default=None,
                        help="vocabulary used for background: basic, wiki")
    parser.add_argument("--post_process", type=str, default=None,
                        help="whether to do post-processing on word embedding")
    config = parser.parse_args()

    # - - - - - - - - - - - - - - - - -
    if config.dist_metric == 'cos':
        config.normalization = True
    elif config.dist_metric == 'l2':
        config.normalization = False
    else:
        sys.exit("Distance Metric NOT SUPPORTED: {}".format(config.dist_metric))
    # - - - - - - - - - - - - - - - - -

    config.eval_type = config.eval_type.split(',')
    config.background_vocab_type = config.background_vocab_type.split(',')
    config.centralization = True

    # display parameters
    logging.info("*** Parameters ***")
    for item, value in vars(config).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")

    # - - - - - - - - - - - - - - - - -
    # load data
    word_pairs_data = w_data_loader.Word_dataset_loader(config)
    # load embedding model
    word_emb_model = w_model.Word_embedding_model(config)
    # evaluation
    our_evaluator = w_evaluation.Word_emb_evaluator(config, word_pairs_data, word_emb_model)
    our_evaluator.eval()