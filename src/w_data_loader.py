#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-19 14:00:56
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

class Word_dataset_loader:
    '''
        dataset loader for word similarity tasks
    '''
    def __init__(self, config) -> None:
        ''' initialization '''

        logging.info("*** Data Preparation ***")
        logging.info("")

        self.config = config

        # load data for word similarity
        if 'similarity' in config.eval_type:
            self.ws_data          = {} # word similarity data
            self.ws_data_path     = './data/word_similarity/'
            self.ws_dataset_names = [
                                'EN-WS-353-ALL.txt',
                                'EN-WS-353-SIM.txt',
                                'EN-WS-353-REL.txt',
                                'EN-MC-30.txt',
                                'EN-RG-65.txt',
                                'EN-RW-STANFORD.txt',
                                'EN-MEN-TR-3k.txt',
                                'EN-MTurk-287.txt',
                                'EN-MTurk-771.txt',
                                'EN-YP-130.txt',
                                'EN-SIMLEX-999.txt',
                                'EN-VERB-143.txt',
                                'EN-SimVerb-3500.txt'
                                ]
            logging.info("Loading {} Word Similarity Datasets".format(len(self.ws_dataset_names)))
            self.load_word_similarity_dataset()
            logging.info("Finished")
            logging.info("")

        if 'ranking' in config.eval_type:
            self.pos_pairs      = [] # ranking: pos pairs
            self.vocab          = [] # ranking: background vocab
            self.rank_data_path = './data/word_evalrank/'

            logging.info("Loading Similar Word Pairs for Ranking")
            self.load_pos_pairs()
            self.pos_pairs.sort()
            logging.info("")

            logging.info("Loading Background Vocab for Ranking")
            self.build_basic_vocab()
            self.build_more_vocab()

    
    def load_word_similarity_dataset(self):
        ''' load word similarity datasets (e.g. {'EN-WS-353-ALL.txt': [['book', 'paper', 5.25]]} '''
        
        for dataset_name in self.ws_dataset_names:
            full_dataset_path = self.ws_data_path + dataset_name
            cur_dataset = []
            with open(full_dataset_path) as f:
                for line in f:
                    x, y, sim_score = line.strip().lower().split()
                    cur_dataset.append([x,y,float(sim_score)])
            self.ws_data[dataset_name] = cur_dataset        


    def load_pos_pairs(self):
        ''' collect positive pairs from word similarity dataset '''

        logging.info("Top 25% from Word Similarity Dataset are used as Positive Pairs")
        
        with open(self.rank_data_path + 'pos_pair_5514.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                cur_line = line.strip().split('\t')
                self.pos_pairs.append(cur_line)

        logging.info("{} Positive Pairs from Word Similarity Datasets".format(len(self.pos_pairs)))


    def build_basic_vocab(self):
        ''' build basic vocabulary from positive pairs '''

        for item in self.pos_pairs:
            if item[0] not in self.vocab: self.vocab.append(item[0])
            if item[1] not in self.vocab: self.vocab.append(item[1])

        logging.info("{} Background Vocab Collected from Similar Word Pairs".format(len(self.vocab)))


    def build_more_vocab(self):
        ''' build more vocabulary from doc '''

        if 'basic' in self.config.background_vocab_type:
            with open(self.rank_data_path + 'basic_vocab.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cur_line = line.strip()
                    if cur_line not in self.vocab: self.vocab.append(cur_line)
            logging.info("{} Background Vocab Collected from Word Similarity Datasets".format(len(self.vocab)))

        if 'wiki' in self.config.background_vocab_type:
            with open(self.rank_data_path + 'wiki_vocab.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cur_line = line.strip()
                    if cur_line not in self.vocab: self.vocab.append(cur_line)
            logging.info("{} Background Vocab Collected from Wiki".format(len(self.vocab)))
