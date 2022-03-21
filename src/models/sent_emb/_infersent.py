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

def embedder_init(self, config):
    ''' initialize for sentence embedding '''

    logging.info("Load InferSent Model")
    from models.sent_emb.infersent_utils import InferSent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PATH_TO_W2V = config.infersent_emb_path
    MODEL_PATH  = config.infersent_model_path

    if config.infersent_version == 'v1':

        V = 1
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        model = InferSent(params_model)
        
        model.load_state_dict(torch.load(MODEL_PATH))
        model.set_w2v_path(PATH_TO_W2V)

        model      = model.to(device)
        self.model = model

    elif config.infersent_version == 'v2':

        V = 2
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        model = InferSent(params_model)
        
        model.load_state_dict(torch.load(MODEL_PATH))
        model.set_w2v_path(PATH_TO_W2V)

        model      = model.to(device)
        self.model = model


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
