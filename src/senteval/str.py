#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   str.py
@Time    :   2021/11/01 16:12:10
@Author  :   bin.wang
@Version :   1.0
'''


from __future__ import absolute_import, division, unicode_literals

import os
import io
import pandas as pd
import numpy as np
import logging

from scipy.stats import spearmanr, pearsonr, kendalltau

from senteval.utils import cosine


class STREval(object):

    def __init__(self, taskpath, seed=1111):
        logging.info('Transfer task: STR')
        self.seed = seed
        self.loadFile(taskpath)

    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        fpath = fpath + 'sem_text_rel_ranked.csv'
        str = pd.read_csv(fpath)

        data = []
        for i in range(5500):
            row = str.loc[i]
            sent1, sent2 = row['Text'].split("\n")
            score = float(row['Score'])
            data.append([sent1,sent2,score])

        sent1, sent2, gs_scores = map(list, zip(*data))
        self.data = (sent1, sent2, gs_scores)
        self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        sys_scores = []
        input1, input2, gs_scores = self.data
        for ii in range(0, len(gs_scores), params.batch_size):
            batch1 = input1[ii:ii + params.batch_size]
            batch2 = input2[ii:ii + params.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = batcher(params, batch1)
                enc2 = batcher(params, batch2)

                for kk in range(enc2.shape[0]):
                    sys_score = self.similarity(enc1[kk], enc2[kk])
                    sys_scores.append(sys_score)

        results = {'pearson': pearsonr(sys_scores, gs_scores)[0],
                    'spearman': spearmanr(sys_scores, gs_scores)[0],
                    'kendall': kendalltau(sys_scores, gs_scores)[0],
                    'nsamples': len(sys_scores)}

        logging.debug('pearson = %.4f, spearman = %.4f, kendall = %.4f' %
                        (
                        results['pearson'],
                        results['spearman'],
                        results['kendall']))

        return results
