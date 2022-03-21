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

import math
import torch
import numpy as np
from tqdm import trange

from transformers import AutoTokenizer, AutoModel


def embedder_init(self, config):
    ''' initialize for sentence embedding '''

    logging.info("SimCSE Model Preparation")

    self.model_name_or_path = config.model_spec
    self.pooling            = 'last'
    self.cache_dir          = './cache'
    self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.tokenizer   = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
    self.simcse_model = AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).to(self.device)

def embedder_infer_all(self, sent_list, normalization, centralization):
    ''' inference to obtain sentence embedding '''

    logging.debug("Compute SimCSE embeddings")

    sent2id        = {}
    sents_embs     = []
    count          = 0
    ex_batch_size  = 64
    ex_max_batches = math.ceil(len(sent_list)/float(ex_batch_size))

    self.simcse_model.eval()
    with torch.no_grad():
        for cur_batch in trange(ex_max_batches, unit="batches", leave=False):
            
            cur_sents = sent_list[cur_batch*ex_batch_size:cur_batch*ex_batch_size+ex_batch_size]

            model_inputs = self.tokenizer(
                cur_sents,
                add_special_tokens = True,
                return_tensors     = 'pt',
                max_length         = 512,
                padding            = 'longest',
                truncation         = True
            ).to(self.device)

            embeddings = self.simcse_model(
                input_ids            = model_inputs['input_ids'],
                attention_mask       = model_inputs['attention_mask'],
                output_hidden_states = True,
                return_dict          = True
                ).pooler_output

            for bt_index in range(len(cur_sents)):
                sent = cur_sents[bt_index]
                if sent not in sent2id:
                    sent2id[sent] = count
                    count   = count + 1
                    ori_emb = embeddings[bt_index].squeeze().cpu().numpy()
                    sents_embs.append(ori_emb)
                else:
                    continue

    sents_embs = np.stack(sents_embs)

    self.sent2id    = sent2id
    self.sents_embs = sents_embs

    if centralization:
        if self.sents_embs is not None:
            self.sents_embs = self.sents_embs - self.sents_embs.mean(axis=0, keepdims=True)

    if normalization:
            self.normalizing_sent_vectors()
