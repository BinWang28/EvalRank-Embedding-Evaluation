#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 16:12:24
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
import numpy as np

class Sent_embedding_model:
    ''' wrapper for sentence embedding models '''
    
    def __init__(self, config) -> None:
        ''' choose embedding method, choose post processing method (gen sentence embedding first and then post-processing) '''
        
        assert config.sent_emb_model != None, "Must specific the sentence embedding model"

        self.config   = config
        self.sent2id  = None
        self.sents_embs  = None

        logging.info("")
        logging.info("*** Model Initialization ***")

        # Model - 1
        if config.sent_emb_model == 'bow':
            from models.sent_emb._bow import embedder_init, embedder_infer_all
        
        # Model - 2
        if config.sent_emb_model == 'bow_pp':
            from models.sent_emb._bow_pp import embedder_init, embedder_infer_all

        # Model - 3
        if config.sent_emb_model == 'infersent':
            from models.sent_emb._infersent import embedder_init, embedder_infer_all

        # Model - customize
        elif config.sent_emb_model == 'customize':
            from models.sent_emb._customize import embedder_init, embedder_infer_all

        else:
            sys.exit("Sentence embedding model NOT SUPPORTED: {}".format(config.sent_emb_model))


        self.embedder_init = embedder_init
        self.embedder_init(self, config)
        self.embedder_infer_all = embedder_infer_all


    def embedder_all(self, sent_list, normalization=True, centralization=False):
        ''' embedding for all'''
        return self.embedder_infer_all(self, sent_list, normalization=normalization, centralization=centralization)


    # general methods = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def embed(self, sents):
        '''
            retrieve original embedding given list of sentences
            input: list of sentences,
            output: embedding matrix array
        '''

        embeddings = []
        for sent in sents:
            assert sent in self.sent2id, "Sentence must be pre-emb for retrieving (original): {}".format(sent)
            sent_emb = self.sents_embs[self.sent2id[sent]]
            embeddings.append(sent_emb)
        embeddings = np.stack(embeddings)

        return embeddings

    
    def normalizing_sent_vectors(self):
        ''' normalizing sentence vectors for both original embedding and processed embeddings '''

        if self.sents_embs is not None:
            self.sents_embs = self.sents_embs / np.linalg.norm(self.sents_embs, axis=1)[:, np.newaxis]

