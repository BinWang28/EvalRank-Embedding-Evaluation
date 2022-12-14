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
import random
import numpy as np
from tqdm import trange

from transformers import AutoTokenizer


def embedder_init(self, config):
    ''' initialize for sentence embedding '''

    logging.info("BERT-flow Model Preparation")

    self.model_name_or_path = config.model_spec
    self.pooling            = config.pooler
    self.cache_dir          = './cache'
    self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.sent_emb.tflow_utils import TransformerGlow

    self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
    self.bertflow = TransformerGlow(self.model_name_or_path, pooling=self.pooling, cache_dir=self.cache_dir).to(self.device)


def embedder_infer_all(self, sent_list, normalization, centralization):
    '''
        inference package for bertflow embedding model (the whole set of sentences)
        this includes training and inferecen
        has to reset the model, everytime new set of data comes
    '''

    from models.sent_emb.tflow_utils import AdamWeightDecayOptimizer

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters= [
        {
            "params": [p for n, p in self.bertflow.glow.named_parameters()  \
                            if not any(nd in n for nd in no_decay)],  
                            # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in self.bertflow.glow.named_parameters()  \
                            if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWeightDecayOptimizer(
        params=optimizer_grouped_parameters, 
        lr=1e-3, 
        eps=1e-6,
    )

    # Important: Remember to shuffle your training data!!! This makes a huge difference!!!
    sentences = sent_list.copy()
    random.shuffle(sentences)

    batch_size = 32
    max_batches = math.ceil(len(sentences)/batch_size)
    num_iterations = max_batches

    logging.debug("Training BERT-FLOW model")

    with trange(num_iterations, unit="batch", leave=False) as tbar:
        for i in tbar:
            iter = i % max_batches
            batch_sentences = sentences[iter*batch_size:(iter+1)*batch_size]

            model_inputs = self.tokenizer(
                batch_sentences,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=512,
                padding='longest',
                truncation=True
            ).to(self.device)

            self.bertflow.train()
            z, loss = self.bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=True)  # Here z is the sentence embedding
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tbar.set_postfix(loss=float(loss))

    # feature extraction using bert-flow
    sent2id    = {}
    sents_embs = []
    count      = 0

    logging.debug("Feature Extraction")
    ex_batch_size = 64
    ex_max_batches = math.ceil(len(sent_list)/float(ex_batch_size))
    self.bertflow.eval()

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

            _, bt_post_emb = self.bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=False) 
            
            # filter out duplicate sentences
            for bt_index in range(len(cur_sents)):
                sent = cur_sents[bt_index]
                if sent not in sent2id:
                    sent2id[sent] = count
                    count += 1
                    post_emb = bt_post_emb[bt_index]
                    post_emb = post_emb.squeeze().cpu().numpy()
                    sents_embs.append(post_emb)
                else:
                    continue

    sents_embs = np.stack(sents_embs)

    self.sent2id        = sent2id
    self.sents_embs = sents_embs

    if centralization:
        if self.sents_embs is not None:
            self.sents_embs = self.sents_embs - self.sents_embs.mean(axis=0, keepdims=True)

    if normalization:
            self.normalizing_sent_vectors()
