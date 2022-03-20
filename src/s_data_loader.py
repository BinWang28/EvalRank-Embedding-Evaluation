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


import io
import os
import math
import logging
import pandas as pd
from tqdm import tqdm,trange


class Sent_ranking_dataset_loader:
    ''' dataset loader for sentence ranking task '''

    def __init__(self, config) -> None:
        ''' class initialization '''
        
        self.pos_pairs = [] # positive sentence pairs
        self.all_sents = [] # background sentences

        logging.info('')
        logging.info('*** Prepare pos sentence pairs for ranking evaluation ***')
        

        import pdb; pdb.set_trace()

        if 'ss' in config.pos_pairs_type:
            self.pos_pair_ratio = config.ss_pos_ratio
            self.data_path = './datasets/sent_sim/STS/STSBenchmark/'
            self.ss_pos_pairs()

        if 'srs' in config.pos_pairs_type:
            self.pos_pair_ratio = config.srs_pos_ratio
            self.data_path = './datasets/sent_sim/'
            self.srs_pos_pairs()

        if 'task' in config.pos_pairs_type:
            self.task_name = config.task_name
            self.task_aug_pos_pairs()

        logging.info("")
        logging.info("Loading Background Sentences for Ranking")
        
        
        self.build_basic_sents()

        if 'ss' in config.background_sent_type:
            self.data_path = './datasets/sent_sim/STS/STSBenchmark/'
            self.aug_ss_sents()

        if 'srs' in config.background_sent_type:
            self.data_path = './datasets/sent_sim/'
            self.aug_srs_sents()

        if 'task' in config.background_sent_type:
            pass
            # TODO


    def load_STSBenchmark_file(self, fpath):
        '''
            loading STS-B files (train, val, test)
        '''

        pair_set = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                pair_set.append([text[5], text[6], float(text[4])])

        return pair_set


    def load_SRS_dataset(self, fpath):
        '''
            loading srs files
        '''
        
        pair_set = []
        str = pd.read_csv(fpath)

        for i in range(5500):
            row = str.loc[i]
            sent1, sent2 = row['Text'].split("\n")
            score = float(row['Score'])
            pair_set.append([sent1, sent2, score])

        return pair_set


    def ss_pos_pairs(self):
        '''
            collect positive pairs from sentence similarity datasets
        '''

        logging.info('Use top {}% as positive pairs for ranking evaluation from ss'.format(int(self.pos_pair_ratio*100)))
        
        pos_pairs, all_pairs = [], []
        
        file_names = ['sts-train.csv', 'sts-dev.csv', 'sts-test.csv']
        for file_name in file_names:
            cur_file_path = self.data_path + file_name
            all_pairs.extend(self.load_STSBenchmark_file(cur_file_path))

        # find high-scoring pairs
        srt_data  = sorted(all_pairs, key=lambda x: x[2], reverse=True)
        pos_pairs = srt_data[0:int(len(srt_data)*self.pos_pair_ratio)]
        pos_pairs = [[item[0], item[1]] for item in pos_pairs]

        for item in pos_pairs:
            if item not in self.pos_pairs: self.pos_pairs.append(item)
            if [item[1], item[0]] not in self.pos_pairs: self.pos_pairs.append([item[1], item[0]])

        logging.info('{} positive pairs collected from STSB dataset'.format(len(self.pos_pairs)))


    def srs_pos_pairs(self):
        '''
            collect positive pairs from SRS datasets
        '''

        logging.info('Use top {}% as positive pairs for ranking evaluation from srs'.format(int(self.pos_pair_ratio*100)))
        cur_file_path = self.data_path + 'sem_text_rel_ranked.csv'

        pos_pairs, all_pairs = [], []
        all_pairs.extend(self.load_SRS_dataset(cur_file_path))

        # find high-scoring pairs
        srt_data  = sorted(all_pairs, key=lambda x: x[2], reverse=True)
        pos_pairs = srt_data[0:int(len(srt_data)*self.pos_pair_ratio)]
        pos_pairs = [[item[0], item[1]] for item in pos_pairs]
        
        for item in pos_pairs:
            if item not in self.pos_pairs: self.pos_pairs.append(item)
            if [item[1], item[0]] not in self.pos_pairs: self.pos_pairs.append([item[1], item[0]])

        logging.info('{} positive pairs collected from SRS dataset'.format(len(self.pos_pairs)))


    def task_aug_pos_pairs_v0(self):
        '''
            augment postive pairs by back-translation
        '''

        # Load sentence data
        task_sents = []

        if self.task_name == 'SST2':
            sentence_path = './datasets/sent_classification/SST/binary/'
        elif self.task_name == 'SST5':
            sentence_path = './datasets/sent_classification/SST/fine/'
        elif self.task_name == 'SICKEntailment':
            sentence_path = './datasets/sent_classification/SICK/'
        else:
            sentence_path = './datasets/sent_classification/'+self.task_name+'/'
        
        with open(sentence_path+'sentences.txt', 'r') as f:
            for line in f: task_sents.append(line.strip())

        bt_sents              = []
        back_trans_sents_path = sentence_path + 'bt_sentences.txt'

        if os.path.isfile(back_trans_sents_path):
            ''' load bt sentences'''
            with open(back_trans_sents_path,'r') as f:
                for line in f:
                    bt_sents.append(line.strip())
        
        else:
            ''' compute bt sentences '''
            logging.info("Back-translation: {}".format(self.task_name))

            # Perform Back Translation if not computed before
            import torch
            from transformers import MarianMTModel, MarianTokenizer

            torch.cuda.empty_cache()

            en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            en_fr_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").cuda()

            fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            fr_en_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").cuda()

            clean_sents = []
            for line in task_sents:
                line = [char for char in line if char in "abcdefghijklmnopqrstuzwxyz1234567890 ,.?!'"]
                line = "".join(line)
                clean_sents.append(line)

            ex_batch_size  = 32
            ex_max_batches = math.ceil(len(clean_sents)/float(ex_batch_size))

            en_fr_model.eval()
            fr_en_model.eval()
            
            with torch.no_grad():
                for cur_batch in trange(ex_max_batches, unit="batches", leave=False):

                    cur_sents = clean_sents[cur_batch*ex_batch_size:cur_batch*ex_batch_size+ex_batch_size]

                    token_input = en_fr_tokenizer(cur_sents, return_tensors="pt", padding=True).to('cuda')
                    translated  = en_fr_model.generate(**token_input)
                    trans_fr    = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

                    token_input = fr_en_tokenizer(trans_fr, return_tensors="pt", padding=True).to('cuda')
                    translated  = fr_en_model.generate(**token_input)
                    trans_en    = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

                    bt_sents.extend(trans_en)

        assert len(task_sents) == len(bt_sents), "Number of original sentences and back-translated ones must match"

        for i in range(len(task_sents)):
            if [task_sents[i],bt_sents[i]] not in self.pos_pairs:
                self.pos_pairs.append([task_sents[i],bt_sents[i]])
            if [bt_sents[i],task_sents[i]] not in self.pos_pairs:
                self.pos_pairs.append([bt_sents[i],task_sents[i]])

        with open(back_trans_sents_path, 'w') as f:
            for sent in bt_sents:
                f.write(sent + '\n')

        logging.info('{} positive pairs collected after back-translation'.format(len(self.pos_pairs)))


    def task_aug_pos_pairs(self):
        '''
            augment postive pairs by paraphrasing
        '''

        # Load sentence data
        task_sents = []

        if self.task_name == 'SST2':
            sentence_path = './datasets/sent_classification/SST/binary/'
        elif self.task_name == 'SST5':
            sentence_path = './datasets/sent_classification/SST/fine/'
        elif self.task_name == 'SICKEntailment':
            sentence_path = './datasets/sent_classification/SICK/'
        else:
            sentence_path = './datasets/sent_classification/'+self.task_name+'/'
        
        with open(sentence_path+'sentences.txt', 'r') as f:
            for line in f: task_sents.append(line.strip())

        para_sents              = []
        para_sents_path = sentence_path + 'para_sentences.txt'

        if os.path.isfile(para_sents_path):
            ''' load para sentences'''
            with open(para_sents_path,'r') as f:
                for line in f:
                    para_sents.append(line.strip())
        
        else:
            ''' compute para sentences '''
            logging.info("Paraphrasing: {}".format(self.task_name))
            
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            model     = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").cuda()
            model.eval()

            with torch.no_grad():
                for sentence in tqdm(task_sents):
                    text =  "paraphrase: " + sentence + " </s>"
            
                    encoding = tokenizer.encode_plus(text, padding=True, return_tensors="pt")
                    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

                    outputs = model.generate(
                        input_ids=input_ids, attention_mask=attention_masks,
                        max_length=256,
                        do_sample=True,
                        top_k=120,
                        top_p=0.90,
                        early_stopping=True,
                        num_return_sequences=1
                    )

                    for output in outputs:
                        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                        para_sents.append(line)
                        #print(sentence)
                        #print(line)

            assert len(task_sents) == len(para_sents), "Number of original sentences and back-translated ones must match"

            with open(para_sents_path, 'w') as f:
                for sent in para_sents:
                    f.write(sent + '\n')

        assert len(task_sents) == len(para_sents), "Number of original sentences and back-translated ones must match"

        #max_from_para = 5000

        for i in range(len(task_sents)):
            if [task_sents[i],para_sents[i]] not in self.pos_pairs:
                self.pos_pairs.append([task_sents[i],para_sents[i]])
            if [para_sents[i],task_sents[i]] not in self.pos_pairs:
                self.pos_pairs.append([para_sents[i],task_sents[i]])

            #if len(self.pos_pairs) > max_from_para:
            #    break

        logging.info('{} positive pairs collected after paraphrase generation'.format(len(self.pos_pairs)))


    def build_basic_sents(self):
        '''
           build basic background sentences from pos pairs
        '''

        for item in self.pos_pairs:
            if item[0] not in self.all_sents: self.all_sents.append(item[0])
            if item[1] not in self.all_sents: self.all_sents.append(item[1])

        logging.info('{} sentences as background sentences'.format(len(self.all_sents)))


    def aug_ss_sents(self):
        '''
            augment background sentences from sentence similarity dataset
        '''

        all_pairs  = []
        file_names = ['sts-train.csv', 'sts-dev.csv', 'sts-test.csv']
        for file_name in file_names:
            cur_file_path = self.data_path + file_name
            all_pairs.extend(self.load_STSBenchmark_file(cur_file_path))

        for item in all_pairs:
            if item[0] not in self.all_sents: self.all_sents.append(item[0])
            if item[1] not in self.all_sents: self.all_sents.append(item[1])
            
        logging.info('{} sentences as background sentences after adding sentence similarity sentences'.format(len(self.all_sents)))

        
    def aug_srs_sents(self):
        '''
            augment background sentences from sentence relatedness dataset
        '''

        all_pairs  = []
        cur_file_path = self.data_path + 'sem_text_rel_ranked.csv'
        all_pairs.extend(self.load_SRS_dataset(cur_file_path))


        for item in all_pairs:
            if item[0] not in self.all_sents: self.all_sents.append(item[0])
            if item[1] not in self.all_sents: self.all_sents.append(item[1])
            
        logging.info('{} sentences as background sentences after adding sentence similarity sentences'.format(len(self.all_sents)))
