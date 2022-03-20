#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 17:23:19
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
from tqdm import tqdm
from prettytable import PrettyTable

import senteval


class Sent_emb_evaluator:
    ''' run evaluation by similarity and ranking '''

    def __init__(self, config, sent_pairs_data, sent_emb_model) -> None:
        ''' initialization for sentence embedder '''

        self.config = config

        self.eval_by_ranking        = 'ranking' in config.eval_type
        self.eval_by_similarity     = 'similarity' in config.eval_type
        self.eval_by_classification = 'classification' in config.eval_type

        self.sent_pairs_data = sent_pairs_data
        self.sent_emb_model  = sent_emb_model

    
    def eval(self):
        ''' main function for evaluation '''

        res_rank = None
        sent_sim = None
        res_cls = None

        if self.eval_by_ranking:
            logging.info('')
            logging.info('*** Evaluation on ranking task ***')
            res_rank = self.eval_for_ranking()

        if self.eval_by_similarity:
            logging.info('')
            logging.info('*** Evaluation on similarity tasks ***')
            sent_sim = self.eval_for_similarity()

        if self.eval_by_classification:
            logging.info('')
            logging.info('*** Evaluation classification tasks ***')
            res_cls = self.eval_for_classification()
        
        return sent_sim, res_rank, res_cls
        

    def prepare(self, params, samples):
        ''' batcher for preparation '''

        samples = [' '.join(sent) if sent != [] else '.' for sent in samples]
        self.sent_emb_model.embedder_all(samples)


    def prepare_nonorm(self, params, samples):
        ''' batcher for preparation '''        

        samples = [' '.join(sent) if sent != [] else '.' for sent in samples]
        self.sent_emb_model.embedder_all(samples, normalization=False)


    def batcher(self, params, batch):
        ''' obtain original sentence embedding given a batch '''

        batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
        embedding = self.sent_emb_model.embed(batch)

        return embedding


    def eval_for_ranking(self):
        ''' evaluate the sentence embeddings with ranking task '''

        hits_max_bound = 15
        res_rank       = None
        
        # pre-compute all embeddings
        self.sent_emb_model.embedder_all(self.sent_pairs_data.all_sents, normalization=self.config.normalization, centralization=True)

        # embedding
        sents_embs = self.sent_emb_model.embed(self.sent_pairs_data.all_sents)
        ranks      = []

        for pair in tqdm(self.sent_pairs_data.pos_pairs):

            s1, s2 = pair
            s1_emb  = self.sent_emb_model.embed([s1])
            s2_emb  = self.sent_emb_model.embed([s2])

            if self.config.dist_metric == 'cos':

                pos_score         = np.dot(s1_emb, s2_emb.T).squeeze()
                background_scores = np.dot(sents_embs, s1_emb.T)
                background_scores = np.squeeze(background_scores)
                background_scores = np.sort(background_scores)[::-1]

            elif self.config.dist_metric == 'l2':

                pos_score         = 1 / (np.linalg.norm(s1_emb - s2_emb) + 1)
                background_scores = 1 / (np.linalg.norm((sents_embs - s1_emb),axis=1) + 1)
                background_scores = np.sort(background_scores)[::-1]

            else:
                sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.config.dist_metric))

            rank = len(background_scores) - np.searchsorted(background_scores[::-1], pos_score, side='right')
            if rank == 0: rank = 1
            ranks.append(int(rank))

        MR  = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        hits_scores = []
        for i in range(hits_max_bound): hits_scores.append(sum(np.array(ranks)<=(i+1))/len(ranks))

        res_rank = {'MR'    : MR,
                    'MRR'   : MRR}

        for i in range(hits_max_bound): res_rank['hits_'+str(i+1)]  = hits_scores[i]

        table = PrettyTable(['Scores', 'Emb'])
        table.add_row(['MR', MR])
        table.add_row(['MRR', MRR])
        for i in range(hits_max_bound): table.add_row(['Hits@'+str(i+1), res_rank['hits_'+str(i+1)]])
        logging.info('Experimental results on ranking')
        logging.info("\n"+str(table))

        return res_rank
        

    def eval_for_similarity(self):
        ''' perform evaluation on similarity tasks '''

        sent_sim = None
        
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'STR']

        # Set params for SentEval
        params_senteval = {'task_path': './data/', 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare_nonorm)
        results = se.eval(transfer_tasks)

        # report result
        table = PrettyTable(['Embs', 'DATASET', 'Pearson', 'Spearman', 'Kendall'])
        for dataset, values in results.items():
            if dataset in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                table.add_row([self.config.sent_emb_model, dataset+'_ALL', results[dataset]['all']['pearson']['wmean'], results[dataset]['all']['spearman']['wmean'], results[dataset]['all']['kendall']['wmean']])
            elif dataset in ['STSBenchmark', 'SICKRelatedness', 'STR']:
                table.add_row([self.config.sent_emb_model, dataset, results[dataset]['pearson'], results[dataset]['spearman'], results[dataset]['kendall']])
        sent_sim = results

        logging.info('Experimental results on similarity for original sentence embeddings')
        logging.info("\n"+str(table))

        return sent_sim
        

    def eval_for_classification(self):
        '''
            evaluate the sentence embedding with classificaition / downstream tasks
        '''
        results = None

        transfer_tasks = ['SCICITE', 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']

        # Set params for SentEval
        params_senteval = {'task_path': './datasets/', 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}

        # evaluation for original embedding and report result
        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare_nonorm)
        results = se.eval(transfer_tasks)

        # results
        logging.info("Classification results on sentence embedding")
        table = PrettyTable(['Dataset', 'SentEmb'])
        for dataset in transfer_tasks:
            table.add_row([dataset, results[dataset]['acc']])
        logging.info("\n"+str(table))

        return results

