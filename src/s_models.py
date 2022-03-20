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
import math
import random
import logging

import numpy as np

from tqdm import tqdm, trange

#import torch
#from sklearn.decomposition import TruncatedSVD
#from transformers.utils.dummy_pt_objects import AutoModel

#from transformers import AutoTokenizer, AutoModel



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
        elif config.sent_emb_model == 'infersent':
            self.infersent_embedder_init(config)
            self.embedder_all = self.infersent_embedder_infer

        elif config.sent_emb_model == 'bert':
            self.bert_embedder_init(config)
            self.embedder_all = self.bert_embedder_infer

        elif config.sent_emb_model == 'bert-flow':
            self.bertflow_embedder_init(config)
            self.embedder_all = self.bertflow_embedder_infer

        #elif config.sent_emb_model == 'bert-whitening':
        #    self.bertwhitening_embedder_init(config)
        #    self.embedder_all = self.bertwhitening_embedder_infer

        elif config.sent_emb_model == 'sbert':
            self.sbert_embedder_init(config)
            self.embedder_all = self.sbert_embedder_infer
            
        elif config.sent_emb_model == 'simcse':
            self.simcse_embedder_init(config)
            self.embedder_all = self.simcse_embedder_infer

        elif config.sent_emb_model == 'customize':
            import pdb; pdb.set_trace()

        else:
            sys.exit("Sentence embedding model NOT SUPPORTED: {}".format(config.sent_emb_model))


        self.embedder_init = embedder_init
        self.embedder_init(self, config)
        self.embedder_infer_all = embedder_infer_all


    def embedder_all(self, sent_list, normalization=True, centralization=False):
        ''' embedding for all'''
        return self.embedder_infer_all(self, sent_list, normalization=normalization, centralization=centralization)


    # InferSent = = = = = = = = = = = = = = = = = = = = = = = = = = = = InferSent (model 39-40)
    def infersent_embedder_init(self, config):
        '''
            initialize for infersent sentence embedding model
        '''

        logging.info("Load InferSent Model")
        from infersent_utils import InferSent

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.infersent_version == 'v1':

            PATH_TO_W2V = './models/glove/glove.840B.300d.txt'
            MODEL_PATH  = './models/infersent/infersent1.pkl'
            V           = 1

            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
            model = InferSent(params_model)
            
            model.load_state_dict(torch.load(MODEL_PATH))
            model.set_w2v_path(PATH_TO_W2V)

            model      = model.to(device)
            self.model = model

        elif config.infersent_version == 'v2':

            PATH_TO_W2V = './models/fasttext/crawl-300d-2M.vec'
            MODEL_PATH  = './models/infersent/infersent2.pkl'
            V           = 2

            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
            model = InferSent(params_model)
            
            model.load_state_dict(torch.load(MODEL_PATH))
            model.set_w2v_path(PATH_TO_W2V)

            model      = model.to(device)
            self.model = model


    def infersent_embedder_infer(self, sent_list, normalization=True, post_process=True, centralization=False):
        '''
            inference to obtain sentence embedding
        '''

        self.model.build_vocab(sent_list, tokenize=False)

        sents_embs_ori = self.model.encode(sent_list, bsize=32, tokenize=False)

        sent2id = {}
        for i in range(len(sent_list)):
            sent2id[sent_list[i]] = i

        self.sent2id        = sent2id
        self.sents_embs_ori = sents_embs_ori

        if post_process:
            assert False, 'No Post-processing Implemented for InferSent'
        else:
            self.sents_embs_post = None

        if centralization:
            if self.sents_embs_ori is not None:
                self.sents_embs_ori = self.sents_embs_ori - self.sents_embs_ori.mean(axis=0, keepdims=True)
            if self.sents_embs_post is not None:
                self.sents_embs_post = self.sents_embs_post - self.sents_embs_post.mean(axis=0, keepdims=True)

        if normalization:
            self.normalizing_sent_vectors()

    # BERT = = = = = = = = = = = = = = = = = = = = = = = = = = = = BERT (model 41 - 60)
    def bert_embedder_init(self, config):
        '''
            initialize for bert-based models
        '''
        logging.info("BERT Model Preparation")

        self.model_name_or_path = config.model_spec
        self.pooling            = config.pooler
        self.cache_dir          = './cache'
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        self.bert_model = AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).to(self.device)


    def bert_embedder_infer(self, sent_list, normalization=True, post_process=True, centralization=False):
        '''
            inference to obtain sentence embedding and post-process
        '''
        logging.debug("Compute BERT features")

        sent2id        = {}
        sents_embs_ori = []
        count          = 0
        ex_batch_size  = 64
        ex_max_batches = math.ceil(len(sent_list)/float(ex_batch_size))
        
        self.bert_model.eval()
        with torch.no_grad():
            for cur_batch in trange(ex_max_batches, unit="batches", leave=False):
            #for cur_batch in range(ex_max_batches):

                cur_sents = sent_list[cur_batch*ex_batch_size:cur_batch*ex_batch_size+ex_batch_size]

                model_inputs = self.tokenizer(
                    cur_sents,
                    add_special_tokens = True,
                    return_tensors     = 'pt',
                    max_length         = 512,
                    padding            = 'longest',
                    truncation         = True
                ).to(self.device)

                all_hidden_states = self.bert_model(
                    input_ids            = model_inputs['input_ids'],
                    attention_mask       = model_inputs['attention_mask'],
                    output_hidden_states = True,
                    return_dict          = True
                    ).hidden_states

                lengths = model_inputs['attention_mask'].sum(dim=1, keepdim=True)  # (bsz, 1)

                if self.pooling == 'cls':
                    bt_ori_emb = all_hidden_states[-1][:,0]
                elif self.pooling == 'last-avg':
                    bt_ori_emb = ((all_hidden_states[-1] * model_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)).div(lengths)  # (bsz, hdim)
                elif self.pooling == 'first-last-avg':
                    bt_ori_emb = ((all_hidden_states[1] * model_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1) + \
                                  (all_hidden_states[-1] * model_inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)
                                ).div(2 * lengths)  # (bsz, hdim)

                for bt_index in range(len(cur_sents)):
                    sent = cur_sents[bt_index]
                    if sent not in sent2id:
                        sent2id[sent] = count
                        count   = count + 1
                        ori_emb = bt_ori_emb[bt_index].squeeze().cpu().numpy()
                        sents_embs_ori.append(ori_emb)
                    else:
                        continue

        sents_embs_ori = np.stack(sents_embs_ori)

        self.sent2id        = sent2id
        self.sents_embs_ori = sents_embs_ori

        if post_process:
            self.bertwhitening_post_process()
        else:
            self.sents_embs_post = None

        if centralization:
            if self.sents_embs_ori is not None:
                self.sents_embs_ori = self.sents_embs_ori - self.sents_embs_ori.mean(axis=0, keepdims=True)
            if self.sents_embs_post is not None:
                self.sents_embs_post = self.sents_embs_post - self.sents_embs_post.mean(axis=0, keepdims=True)

        if normalization:
            self.normalizing_sent_vectors()


    # BERT-Whitening = = = = = = = = = = = = = = = = = = = = = = = = = = = = BERT-Whitening
    def bertwhitening_post_process(self):
        '''
            perform post-processing for bertwhitening model
        '''

        assert self.sents_embs_ori.any() != None, "The original embeddings should be computed fist before post-processing"

        logging.debug("Perform Whitening")

        all_embs = self.sents_embs_ori
        mean     = all_embs.mean(axis=0, keepdims=True)
        cov      = np.cov(all_embs.T)
        u, s, _  = np.linalg.svd(cov)
        W        = np.dot(u, np.diag(np.sqrt(1/s)))

        new_embs = (all_embs-mean).dot(W)

        self.sents_embs_post = new_embs


    # bert-flow = = = = = = = = = = = = = = = = = = = = = = = = = = = = bert-flow (model 61-63)
    # https://arxiv.org/abs/2011.05864
    def bertflow_embedder_init(self, config):
        '''
            initialize for bert-flow sentence embedding
            we perform training and extraction at the embedder all part
        '''

        logging.info("BERT-flow Model Preparation")

        self.model_name_or_path = config.model_spec
        self.pooling            = config.pooler
        self.cache_dir          = './cache'
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from tflow_utils import TransformerGlow

        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        self.bertflow = TransformerGlow(self.model_name_or_path, pooling=self.pooling, cache_dir=self.cache_dir).to(self.device)


    def bertflow_embedder_infer(self, sent_list, normalization=True, post_process=True, centralization=False):
        '''
            inference package for bertflow embedding model (the whole set of sentences)
            this includes training and inferecen
            has to reset the model, everytime new set of data comes
        '''

        from tflow_utils import AdamWeightDecayOptimizer

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters= [
            {
                "params": [p for n, p in self.bertflow.glow.named_parameters()  \
                                if not any(nd in n for nd in no_decay)],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
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
        sent2id         = {}
        sents_embs_ori  = []
        sents_embs_post = []
        count           = 0

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
                        sents_embs_post.append(post_emb)
                    else:
                        continue

        sents_embs_post = np.stack(sents_embs_post)

        self.sent2id        = sent2id
        self.sents_embs_ori = sents_embs_post

        assert post_process==False, "There is no post-processing for BERT-flow model"
        self.sents_embs_post = None

        if centralization:
            if self.sents_embs_ori is not None:
                self.sents_embs_ori = self.sents_embs_ori - self.sents_embs_ori.mean(axis=0, keepdims=True)
            if self.sents_embs_post is not None:
                self.sents_embs_post = self.sents_embs_post - self.sents_embs_post.mean(axis=0, keepdims=True)

        if normalization:
            self.normalizing_sent_vectors() # normalization for both original and post-processed vectors


    # SBERT = = = = = = = = = = = = = = = = = = = = = = = = = = = = SBERT (model 64, 65)
    # https://arxiv.org/abs/1908.10084
    def sbert_embedder_init(self, config):
        '''
            initialize for SBERT sentence embedding model
        '''
        
        #self.model_name_or_path = 'sentence-transformers/bert-base-nli-mean-tokens'
        self.model_name_or_path = config.model_spec
        self.pooling            = 'last'
        self.cache_dir          = './cache'
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("The post-processing of SBERT is using BERT-Whitening approach")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        self.sbert_model = AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).to(self.device)


    def sbert_embedder_infer(self, sent_list, normalization=True, post_process=True, centralization=False):
        '''
            inference to obtain sentence embedding 
        '''

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        logging.debug("Compute SBERT embeddings")

        sent2id        = {}
        sents_embs_ori = []
        count          = 0
        ex_batch_size  = 64
        ex_max_batches = math.ceil(len(sent_list)/float(ex_batch_size))

        self.sbert_model.eval()
        with torch.no_grad():
            #for cur_batch in trange(ex_max_batches, unit="batches", leave=False):
            for cur_batch in range(ex_max_batches):

                cur_sents = sent_list[cur_batch*ex_batch_size:cur_batch*ex_batch_size+ex_batch_size]

                model_inputs = self.tokenizer(
                    cur_sents,
                    add_special_tokens = True,
                    return_tensors     = 'pt',
                    max_length         = 512,
                    padding            = 'longest',
                    truncation         = True
                ).to(self.device)

                embeddings = self.sbert_model(
                    input_ids            = model_inputs['input_ids'],
                    attention_mask       = model_inputs['attention_mask'],
                    output_hidden_states = True,
                    return_dict          = True
                    )

                embeddings = mean_pooling(embeddings, model_inputs['attention_mask'])

                for bt_index in range(len(cur_sents)):
                    sent = cur_sents[bt_index]
                    if sent not in sent2id:
                        sent2id[sent] = count
                        count   = count + 1
                        ori_emb = embeddings[bt_index].squeeze().cpu().numpy()
                        sents_embs_ori.append(ori_emb)
                    else:
                        continue

        sents_embs_ori = np.stack(sents_embs_ori)

        self.sent2id        = sent2id
        self.sents_embs_ori = sents_embs_ori

        if post_process:
            self.bertwhitening_post_process()
        else:
            self.sents_embs_post = None

        if centralization:
            if self.sents_embs_ori is not None:
                self.sents_embs_ori = self.sents_embs_ori - self.sents_embs_ori.mean(axis=0, keepdims=True)
            if self.sents_embs_post is not None:
                self.sents_embs_post = self.sents_embs_post - self.sents_embs_post.mean(axis=0, keepdims=True)

        if normalization:
            self.normalizing_sent_vectors()


    # SimCSE = = = = = = = = = = = = = = = = = = = = = = = = = = = = SimCSE (model 66-67)
    # https://arxiv.org/abs/2104.08821
    def simcse_embedder_init(self, config):
        '''
            initialize for SimCSE sentence embedding model
        '''
        #self.model_name_or_path = 'princeton-nlp/unsup-simcse-bert-base-uncased'
        self.model_name_or_path = config.model_spec
        self.pooling            = 'last'
        self.cache_dir          = './cache'
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        self.simcse_model = AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).to(self.device)

        #tokenizer.save_pretrained('./models/unsup-simcse-bert-base-uncased')
        #simcse_model.save_pretrained('./models/unsup-simcse-bert-base-uncased')


    def simcse_embedder_infer(self, sent_list, normalization=True, post_process=True, centralization=False):
        '''
            inference to obtain sentence embedding (no post processing model available)
        '''

        logging.debug("Compute SimCSE embeddings")

        sent2id        = {}
        sents_embs_ori = []
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
                        sents_embs_ori.append(ori_emb)
                    else:
                        continue
        sents_embs_ori = np.stack(sents_embs_ori)

        self.sent2id        = sent2id
        self.sents_embs_ori = sents_embs_ori

        assert post_process==False, "There is no post-processing for SimCSE model"
        self.sents_embs_post = None

        if centralization:
            if self.sents_embs_ori is not None:
                self.sents_embs_ori = self.sents_embs_ori - self.sents_embs_ori.mean(axis=0, keepdims=True)
            if self.sents_embs_post is not None:
                self.sents_embs_post = self.sents_embs_post - self.sents_embs_post.mean(axis=0, keepdims=True)

        if normalization:
            self.normalizing_sent_vectors()


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

