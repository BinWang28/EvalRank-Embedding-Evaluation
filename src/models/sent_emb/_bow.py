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
import copy

import numpy as np

from w_model import Word_embedding_model


def embedder_init(self, config):
    ''' initialize for bow sentence embedding '''

    config = copy.deepcopy(config) # to not influence other

    assert config.bow_we_path != None, "Must specific the word embedding path if using BOW model"

    logging.info('BOW sentence embedding')

    config.word_emb_model   = config.bow_we_path
    config.normalization    = False # for word embedding normalization
    config.post_process     = 'False' # for word embedding post-processing
    config.centralization   = False # for word embedding centralizing
    self.our_word_emb_model = Word_embedding_model(config)


def embedder_infer_all(self, sent_list, normalization, centralization):
    ''' inference package for bow embedding for all needed sentences '''

    sent2id    = {}
    sents_embs = []
    count      = 0

    for sent in sent_list:
        # skip if already computed
        if sent not in sent2id:
            sent2id[sent] = count
            count += 1
        else:
            continue

        # skip words not in vocab
        # use zero vector for unknown sents
        sent_split = sent.lower().split()

        sentvec = []
        for word in sent_split:
            if word in self.our_word_emb_model.vocab:
                sentvec.append(self.our_word_emb_model.compute_embedding(word))
            else:
                continue
        
        # if not words are found, use zeros as the representation
        if not sentvec:
            vec = np.zeros(self.our_word_emb_model.wvec_dim) + 1e-9
            sentvec.append(vec)
            
        sentvec = np.mean(sentvec, 0)
        sents_embs.append(sentvec)
   
    sents_embs = np.stack(sents_embs)

    self.sent2id    = sent2id
    self.sents_embs = sents_embs

    if centralization:
        if self.sents_embs is not None:
            self.sents_embs = self.sents_embs - self.sents_embs.mean(axis=0, keepdims=True)

    if normalization:
            self.normalizing_sent_vectors()