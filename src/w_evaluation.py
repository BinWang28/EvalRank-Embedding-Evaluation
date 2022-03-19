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
        
        self.eval_by_similarity = 'similarity' in config.eval_type
        self.eval_by_ranking    = 'ranking' in config.eval_type
        self.dist_metric        = config.dist_metric

        self.word_pairs_data = word_pairs_data
        self.word_emb_model  = word_emb_model

    def eval(self):
        ''' main functions for this module '''
        
        if self.eval_by_similarity:

            logging.info('')
            logging.info('*** Evaluation on word similarity tasks ***')
            table_ws, res_ws = self.eval_for_similarity()
            logging.info("\n"+str(table_ws))

        if self.eval_by_ranking:

            logging.info('')
            logging.info('*** Evaluation on word ranking tasks ***')
            table_rank, res_rank = self.eval_for_ranking()
            logging.info("\n"+str(table_rank))
        
        return res_ws, res_rank

    
    def eval_for_similarity(self):
        ''' evaluate the embeddings (ori, post) on similarity task '''

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
            table.add_row([count, 'Ori Emb', dataset, resu['Pearsaon Corr'], resu['Spearman Corr'], resu['Kendall Corr']])
            count += 1
        
        return table, results


    def eval_for_ranking(self):
        ''' evaluate the embeddings (ori, post) on ranking task '''

        vocab_embs_ori  = [self.word_emb_model.compute_embedding(word)[0] for word in self.word_pairs_data.vocab]
        vocab_embs_ori  = np.stack(vocab_embs_ori)
        vocab_embs_post = [self.word_emb_model.compute_embedding(word)[1] for word in self.word_pairs_data.vocab]
        vocab_embs_post = np.stack(vocab_embs_post)

        ranks_ori  = []
        ranks_post = []

        if self.dist_metric == 'mahalanobis': pre_w1 = None

        #for pair in tqdm(self.word_pairs_data.pos_pairs, leave=False):
        for pair in self.word_pairs_data.pos_pairs:

            w1        , w2          = pair
            w1_emb_ori, w1_emb_post = self.word_emb_model.compute_embedding(w1)
            w2_emb_ori, w2_emb_post = self.word_emb_model.compute_embedding(w2)

            # results for ori & post embeddings
            if self.dist_metric == 'cos' or self.dist_metric == 'inner':
            
                pos_score_ori         = np.dot(w1_emb_ori, w2_emb_ori)
                background_scores_ori = np.dot(vocab_embs_ori, w1_emb_ori)
                background_scores_ori = np.sort(background_scores_ori)[::-1]

                pos_score_post         = np.dot(w1_emb_post, w2_emb_post)
                background_scores_post = np.dot(vocab_embs_post, w1_emb_post)
                background_scores_post = np.sort(background_scores_post)[::-1]
            
            elif self.dist_metric == 'l2':
            
                pos_score_ori         = 1 / (np.linalg.norm(w1_emb_ori - w2_emb_ori) + 1)
                background_scores_ori = 1 / (np.linalg.norm((vocab_embs_ori - w1_emb_ori),axis=1) + 1)
                background_scores_ori = np.sort(background_scores_ori)[::-1]

                pos_score_post         = 1 / (np.linalg.norm(w1_emb_post - w2_emb_post) + 1)
                background_scores_post = 1 / (np.linalg.norm((vocab_embs_post - w1_emb_post),axis=1) + 1)
                background_scores_post = np.sort(background_scores_post)[::-1]

            else: 
                sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.dist_metric))
            
            
            rank_ori = len(background_scores_ori) - np.searchsorted(background_scores_ori[::-1], pos_score_ori, side='right')
            if rank_ori == 0: rank_ori = 1
            ranks_ori.append(int(rank_ori))

            rank_post = len(background_scores_post) - np.searchsorted(background_scores_post[::-1], pos_score_post, side='right')
            if rank_post == 0: rank_post = 1
            ranks_post.append(int(rank_post))


        MR_ori   = np.mean(ranks_ori)
        MRR_ori  = np.mean(1. / np.array(ranks_ori))
        MR_post  = np.mean(ranks_post)
        MRR_post = np.mean(1. / np.array(ranks_post))

        hits_scores_ori  = []
        hits_scores_post = []
        hits_max_bound   = 15
        
        for i in range(hits_max_bound):
            hits_scores_ori.append(sum(np.array(ranks_ori)<=(i+1))/len(ranks_ori))
            hits_scores_post.append(sum(np.array(ranks_post)<=(i+1))/len(ranks_post))

        res_rank = {
                    'MR_ori'    : MR_ori,
                    'MRR_ori'   : MRR_ori,
                    'MR_post'    : MR_post,
                    'MRR_post'   : MRR_post}

        for i in range(hits_max_bound):
            res_rank['hits_'+str(i+1)+'_ori']  = hits_scores_ori[i]
            res_rank['hits_'+str(i+1)+'_post'] = hits_scores_post[i]


        table = PrettyTable(['Scores', 'Ori Emb', 'Post Emb'])
        table.add_row(['MR',         MR_ori,       MR_post])
        table.add_row(['MRR',        MRR_ori,      MRR_post])
        
        for i in range(hits_max_bound):
            table.add_row(['Hits@'+str(i+1), res_rank['hits_'+str(i+1)+'_ori'], res_rank['hits_'+str(i+1)+'_post']])

        return table, res_rank
