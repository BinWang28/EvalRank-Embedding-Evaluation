#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 16:44:31
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import logging
import torch

from transformers import AutoTokenizer, AutoModel


def embedder_init(self, config):
    ''' initialize for sentence embedding '''

    logging.info("BERT Model Preparation")
    self.model_name_or_path = config.model_spec
    self.pooling            = config.pooler
    self.cache_dir          = './cache'
    self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
    self.bert_model = AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).to(self.device)



def embedder_infer_all(self, sent_list, normalization, centralization):
    ''' inference package for embedding for all needed sentences '''

    self.model.build_vocab(sent_list, tokenize=False)

    sents_embs = self.model.encode(sent_list, bsize=32, tokenize=False)

    sent2id = {}
    for i in range(len(sent_list)):
        sent2id[sent_list[i]] = i

    self.sent2id    = sent2id
    self.sents_embs = sents_embs

    if centralization:
        if self.sents_embs is not None:
            self.sents_embs = self.sents_embs - self.sents_embs.mean(axis=0, keepdims=True)

    if normalization:
            self.normalizing_sent_vectors()
