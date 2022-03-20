#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 15:39:12
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

class Sent_ranking_dataset_loader:
    ''' dataset loader for sentence ranking task '''

    def __init__(self, config) -> None:
        ''' class initialization '''
        
        self.pos_pairs = [] # positive sentence pairs
        self.all_sents = [] # background sentences

        logging.info('')
        logging.info('*** Prepare pos sentence pairs for ranking evaluation ***')
        
        self.rank_data_path = './data/sent_evalrank/'

        with open(self.rank_data_path + 'pos_pair.txt', 'r') as f: 
            lines = f.readlines()
            for line in lines:
                sent1, sent2 = line.strip().split('\t')
                self.pos_pairs.append([sent1, sent2])
        logging.info('{} positive pairs collected from STSB dataset'.format(len(self.pos_pairs)))


        logging.info("")
        logging.info("Loading Background Sentences for Ranking")
        
        self.build_basic_sents()

        with open(self.rank_data_path + 'background_sent.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line not in self.all_sents:
                    self.all_sents.append(line)

        logging.info('{} sentences as background sentences'.format(len(self.all_sents)))


    def build_basic_sents(self):
        ''' build basic background sentences from pos pairs '''

        for item in self.pos_pairs:
            if item[0] not in self.all_sents: self.all_sents.append(item[0])
            if item[1] not in self.all_sents: self.all_sents.append(item[1])

        logging.info('{} sentences as background sentences'.format(len(self.all_sents)))
