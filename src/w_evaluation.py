#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-19 16:45:21
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

from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau

class Word_emb_evaluator:
    ''' run evaluation by similarity and ranking '''
    def __init__(self, config, word_pairs_data, word_emb_model) -> None:
        ''' define what tasks to perform in this module '''

        self.eval_by_ranking    = 'ranking' in config.eval_type
        self.eval_by_similarity = 'similarity' in config.eval_type
        self.dist_metric        = config.dist_metric

        self.word_pairs_data = word_pairs_data
        self.word_emb_model  = word_emb_model


    def eval(self):
        ''' main functions for this module '''
        
        if self.eval_by_ranking:

            logging.info('')
            logging.info('*** Evaluation on word ranking tasks ***')
            table_rank, res_rank = self.eval_for_ranking()
            logging.info("\n"+str(table_rank))

        if self.eval_by_similarity:

            logging.info('')
            logging.info('*** Evaluation on word similarity tasks ***')
            table_ws, res_ws = self.eval_for_similarity()
            logging.info("\n"+str(table_ws))
       
        return res_ws, res_rank

    
    def eval_for_ranking(self):
        ''' evaluate the embeddings on ranking task '''

        ranks      = []
        vocab_embs = [self.word_emb_model.compute_embedding(word) for word in self.word_pairs_data.vocab]
        vocab_embs = np.stack(vocab_embs)

        for pair in tqdm(self.word_pairs_data.pos_pairs, leave=False):
        # for pair in self.word_pairs_data.pos_pairs:

            w1, w2 = pair
            w1_emb = self.word_emb_model.compute_embedding(w1)
            w2_emb = self.word_emb_model.compute_embedding(w2)

            if self.dist_metric == 'cos':
            
                pos_score         = np.dot(w1_emb, w2_emb)
                background_scores = np.dot(vocab_embs, w1_emb)
                background_scores = np.sort(background_scores)[::-1]
            
            elif self.dist_metric == 'l2':
            
                pos_score         = 1 / (np.linalg.norm(w1_emb - w2_emb) + 1)
                background_scores = 1 / (np.linalg.norm((vocab_embs - w1_emb),axis=1) + 1)
                background_scores = np.sort(background_scores)[::-1]

            else: 
                sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.dist_metric))
            
            rank = len(background_scores) - np.searchsorted(background_scores[::-1], pos_score, side='right')

            if rank == 0: rank = 1
            ranks.append(int(rank))

        MR  = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        hits_scores    = []
        hits_max_bound = 15
        
        for i in range(hits_max_bound):
            hits_scores.append(sum(np.array(ranks)<=(i+1))/len(ranks))

        res_rank = {'MR'    : MR,
                    'MRR'   : MRR}

        for i in range(hits_max_bound): res_rank['hits_'+str(i+1)]  = hits_scores[i]

        table = PrettyTable(['Scores', 'Emb'])
        table.add_row(['MR', MR])
        table.add_row(['MRR', MRR])
        
        for i in range(hits_max_bound):
            if i in [0,2]:
                table.add_row(['Hits@'+str(i+1), res_rank['hits_'+str(i+1)]])

        return table, res_rank


    def eval_for_similarity(self):
        ''' evaluate the embeddings on similarity task '''

        results  = {}

        for dataset_name, data_pairs in tqdm(self.word_pairs_data.ws_data.items(), leave=False): 
            predicts = []
            expected = []

            for w1, w2, sc in data_pairs: 

                w1_emb = self.word_emb_model.compute_embedding(w1)
                w2_emb = self.word_emb_model.compute_embedding(w2)

                if self.dist_metric == 'cos':
                    predict  = w1_emb.dot(w2_emb.transpose())

                elif self.dist_metric == 'l2':
                    predict  = 1 / (np.linalg.norm(w1_emb - w2_emb) + 1) # note 1/(1+d)
                    
                else:
                    sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.dist_metric))
                
                predicts.append(predict)                
                expected.append(sc)
            
            pearsonr_res  = pearsonr(predicts, expected)[0]
            spearmanr_res = spearmanr(predicts, expected)[0]
            kendall_res   = kendalltau(predicts, expected)[0]

            results[dataset_name] = {'Pearsaon Corr': pearsonr_res,
                                     'Spearman Corr': spearmanr_res,
                                     'Kendall Corr' : kendall_res}
        
        table = PrettyTable(['Index', 'METHOD', 'DATASET', 'Pearson', 'Spearman', 'Kendall'])
        count = 1
        for dataset, resu in results.items():
            table.add_row([count, 'Emb', dataset, resu['Pearsaon Corr'], resu['Spearman Corr'], resu['Kendall Corr']])
            count += 1
        
        return table, results