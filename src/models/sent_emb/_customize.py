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

import copy
import logging

import numpy as np
from tqdm import tqdm

# customize your own sentence embedding model

def embedder_init(self, config):
    ''' initialize for bow sentence embedding '''

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # sentence embedding model initialization if needed
    #


        
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *



def embedder_infer_all(self, sent_list, normalization, centralization):
    ''' inference package for bow embedding for all needed sentences '''

    sent2id    = {}
    sents_embs = []
    count      = 0

    for sent in tqdm(sent_list, leave=False):
        # skip if already computed
        if sent not in sent2id:
            sent2id[sent] = count
            count += 1
        else:
            continue
        
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # sentvec = your model output vector for current sentence
        #



        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        sents_embs.append(sentvec)
   
    sents_embs = np.stack(sents_embs)

    self.sent2id    = sent2id
    self.sents_embs = sents_embs

    if centralization:
        if self.sents_embs is not None:
            self.sents_embs = self.sents_embs - self.sents_embs.mean(axis=0, keepdims=True)

    if normalization:
            self.normalizing_sent_vectors()