###
# File: Untitled-1
# Project: <<projectpath>>
# Created Date: 2022-03-19 14:00:49
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


#import io
#import os
#import re
#import json
#import math
#import string
import logging
#import pandas as pd
#from tqdm import tqdm,trange



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

class Word_dataset_loader:
    '''
        dataset loader for word similarity tasks
    '''
    def __init__(self, config) -> None:
        '''
            class initialization
        '''

        logging.info("*** Data Preparation ***")
        logging.info("")

        if 'similarity' in config.eval_type:
            self.ws_data          = {} # word similarity data
            self.ws_data_path     = './data/word/similarity/'
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

            logging.info("Loading {} Word Similarity Datasets".format(len(self.dataset_names)))

            self.load_word_similarity_dataset()




        if 'ranking' in config.eval_type:
            self.pos_pairs = [] # ranking: pos pairs
            self.vocab     = [] # ranking: background vocab
        


        if self.pos_pairs_type is not None:
            logging.info("")
            logging.info("Loading Similar Word Pairs for Ranking")
            if 'ws' in self.pos_pairs_type:
                self.ws_pos_ratio = config.ws_pos_ratio
                self.ws_pos_pairs()
            if 'synonym' in self.pos_pairs_type:
                self.synonym_freq_num = config.synonym_freq_num
                self.synonym_pos_pairs()
            if 'task' in self.pos_pairs_type:
                self.task_name = config.task_name
                self.task_pos_pairs_filter()
            if 'task2' in self.pos_pairs_type:
                self.task_name = config.task_name
                self.task_aug_pos_pairs()

            logging.info("")
            logging.info("Loading Background Vocab for Ranking")
            self.build_basic_vocab()
        
        if self.background_vocab_type is not None:
            if 'ws' in self.background_vocab_type:
                self.aug_ws_vocab()
            if 'task' in self.background_vocab_type:
                self.task_name = config.task_name
                self.aug_task_vocab()
            if 'task2' in self.background_vocab_type:
                # TODO
                pass
            if 'wiki_vocab' in self.background_vocab_type:
                self.aug_wiki_vocab()

        # sort pairs for speeding up
        self.pos_pairs.sort()

        #with open('pos_pair_5514.txt', 'w') as f:
        #    for item in self.pos_pairs:
        #        f.write(item[0]+ '\t' + item[1] + '\n')

        #with open('wiki_vocab.txt', 'w') as f:
        #    for item in self.vocab:
        #        f.write(item + '\n')


        #import pdb; pdb.set_trace()

    
    def load_word_similarity_dataset(self):
        '''
            load word similarity textual dataset, (e.g. {'EN-WS-353-ALL.txt': [['book', 'paper', 5.25]]}
        '''
        
        for dataset_name in self.dataset_names:
            full_dataset_path = self.data_path + dataset_name
            cur_dataset = []
            with open(full_dataset_path) as f:
                for line in f:
                    x, y, sim_score = line.strip().lower().split()
                    cur_dataset.append([x,y,float(sim_score)])

            self.ws_data[dataset_name] = cur_dataset        

        
    def ws_pos_pairs(self):
        '''
            collect positive pairs from word similarity dataset
        '''

        logging.info("Use Top {}% from Word Similarity Dataset as Positive Pairs".format(int(self.ws_pos_ratio*100)))

        #dataset2collect = [
        #                    'EN-VERB-143.txt',
        #                    'EN-RW-STANFORD.txt',
        #                    'EN-MTurk-771.txt',
        #                    'EN-MTurk-287.txt',
        #                    ]

        for dataset_name in self.dataset_names:
            #if dataset_name not in dataset2collect:
            #    continue

            cur_dataset   = self.ws_data[dataset_name]
            srt_data      = sorted(cur_dataset, key=lambda x: x[2], reverse=True)
            cur_pos_pairs = srt_data[0:int(len(srt_data)*self.ws_pos_ratio)] # keep top xx percent
            cur_pos_pairs = [[item[0], item[1]] for item in cur_pos_pairs]
            cur_pos_pairs = [item for item in cur_pos_pairs if item[0]!=item[1]]

            # add w1-w2 and w2-w1
            for item in cur_pos_pairs:
                if item not in self.pos_pairs:
                    self.pos_pairs.append(item)
                if [item[1], item[0]] not in self.pos_pairs:
                    self.pos_pairs.append([item[1], item[0]])

        logging.info("{} Positive Pairs Collected after Word Similarity Datasets".format(len(self.pos_pairs)))


    def synonym_pos_pairs(self):
        '''
            collect similar word pairs from synonym dataset
        '''

        logging.info("Loading Similar Word Pairs from Synonym Dataset")
        
        def contain_only_en(my_str):
            symbols = [item for item in string.ascii_lowercase]
            return all(char in symbols for char in my_str)

        pos_pairs     = []
        syn_file_name = 'synonyms.json'

        with open(self.data_path + syn_file_name, 'r') as f:
            raw_syn_data = json.load(f)

        # collect word frequency
        wiki_word2freqr      = {}
        self.wfreq_file_name = 'enwiki_vocab_min200.txt'

        with open(self.data_path + self.wfreq_file_name, 'r') as f:
            count = 1
            for line in f:
                wiki_word2freqr[line.split()[0]] = count
                count += 1

        #for key, value in tqdm(raw_syn_data.items(), leave=False):
        for key, value in raw_syn_data.items():
            key   = key.split(':')[0]
            value = re.split(';|\|', value)
            if not contain_only_en(key) or key not in wiki_word2freqr:
                continue
            value = [word for word in value if contain_only_en(word)]
            value = sorted(set(value))
            for item in value:
                if key == item or item not in wiki_word2freqr:
                    continue             
                pos_pair = [key, item]
                if pos_pair not in pos_pairs:
                    pos_pairs.append(pos_pair)
        
        #with open('synonym.txt', 'w') as f:
        #    for item in pos_pairs:
        #        f.write(item[0]+'\t'+item[1]+'\n')
        #import pdb; pdb.set_trace()
        

        # compute threhold
        freqr_threshold = []
        for pair in pos_pairs:
            freqr_threshold.append(wiki_word2freqr[pair[0]]+wiki_word2freqr[pair[1]])
        freqr_threshold.sort()
        freqr_threshold = freqr_threshold[self.synonym_freq_num]

        for item in pos_pairs:
            if (wiki_word2freqr[item[0]] + wiki_word2freqr[item[1]]) > freqr_threshold:
                continue
            if item not in self.pos_pairs:
                self.pos_pairs.append(item)
            if [item[1],item[0]] not in self.pos_pairs:
                self.pos_pairs.append([item[1],item[0]])

        logging.info("{} Positive Pairs Collected after Word Synonym Dataset".format(len(self.pos_pairs)))


    def task_pos_pairs_filter(self):
        '''
            filter similar word pair by task vocabularies
        '''

        new_pos_pairs = []
        task_vocab    = []

        if self.task_name == 'SST2':
            with open('./datasets/sent_classification/SST/binary/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])
        elif self.task_name == 'SST5':
            with open('./datasets/sent_classification/SST/fine/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])
        elif self.task_name == 'SICKEntailment':
            with open('./datasets/sent_classification/SICK/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])
        else:
            with open('./datasets/sent_classification/'+self.task_name+'/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])

        for item in self.pos_pairs:
            if item[0] in task_vocab and item[1] in task_vocab:
                new_pos_pairs.append(item)
        
        self.pos_pairs = new_pos_pairs
        logging.info("{} Positive Pairs Collected after Filtered with Task Vocab".format(len(self.pos_pairs)))


    def task_aug_pos_pairs_v0(self):
        '''
            generate positive word pairs with paraphrase model
        '''

        # load vocab data
        task_vocab = []
        freq_vocab  = []

        if self.task_name == 'SST2':
            sentence_path = './datasets/sent_classification/SST/binary/'
        elif self.task_name == 'SST5':
            sentence_path = './datasets/sent_classification/SST/fine/'
        elif self.task_name == 'SICKEntailment':
            sentence_path = './datasets/sent_classification/SICK/'
        else:
            sentence_path = './datasets/sent_classification/'+self.task_name+'/'
        
        with open(sentence_path+'vocab.txt', 'r') as f:
            for line in f: task_vocab.append(line.strip())

        with open('./datasets/word_sim/enwiki_vocab_min200.txt', 'r') as f:
            for line in f: freq_vocab.append(line.strip().split()[0])

        para_vocab      = []
        para_vocab_path = sentence_path + 'para_vocab.txt'

        if os.path.isfile(para_vocab_path):
            ''' load para vocabs'''
            with open(para_vocab_path,'r') as f:
                for line in f:
                    para_vocab.append(line.strip())

        else:
            ''' compute para vocabs '''
            logging.info("Paraphrasing: {}".format(self.task_name))
            
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            model     = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").cuda()
            model.eval()

            with torch.no_grad():
                for word in tqdm(task_vocab):
                    text =  "paraphrase: " + word + " </s>"
            
                    encoding = tokenizer.encode_plus(text, padding=True, return_tensors="pt")
                    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

                    outputs = model.generate(
                        input_ids=input_ids, attention_mask=attention_masks,
                        max_length=256,
                        do_sample=True,
                        top_k=120,
                        top_p=0.2,
                        early_stopping=True,
                        num_return_sequences=1
                    )

                    for output in outputs:
                        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                        para_vocab.append(line)
                        print(word)
                        print(line)

                        import pdb; pdb.set_trace()
                        # seems not working well

            assert len(task_vocab) == len(para_vocab), "Number of original sentences and back-translated ones must match"

            with open(para_vocab_path, 'w') as f:
                for sent in para_vocab:
                    f.write(sent + '\n')

        assert len(task_vocab) == len(para_vocab), "Number of original sentences and back-translated ones must match"


        pdb.set_trace()


    def task_aug_pos_pairs(self):
        '''
            generate positive word pairs with back-trans model
        '''

        # load vocab data
        task_vocab = []
        freq_vocab  = []

        if self.task_name == 'SST2':
            sentence_path = './datasets/sent_classification/SST/binary/'
        elif self.task_name == 'SST5':
            sentence_path = './datasets/sent_classification/SST/fine/'
        elif self.task_name == 'SICKEntailment':
            sentence_path = './datasets/sent_classification/SICK/'
        else:
            sentence_path = './datasets/sent_classification/'+self.task_name+'/'
        
        with open(sentence_path+'vocab.txt', 'r') as f:
            for line in f: task_vocab.append(line.strip())

        with open('./datasets/word_sim/enwiki_vocab_min200.txt', 'r') as f:
            for line in f: freq_vocab.append(line.strip().split()[0])

        bt_vocab      = []
        bt_vocab_path = sentence_path + 'bt_vocab.txt'

        if os.path.isfile(bt_vocab_path):
            ''' load bt vocabs'''
            with open(bt_vocab_path,'r') as f:
                for line in f:
                    bt_vocab.append(line.strip())

        else:
            ''' compute bt vocabs '''
            logging.info("Back-translation: {}".format(self.task_name))
            
            import torch
            from transformers import MarianMTModel, MarianTokenizer

            en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            en_fr_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").cuda()

            fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            fr_en_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").cuda()

            en_fr_model.eval()
            fr_en_model.eval()

            ex_batch_size  = 32
            ex_max_batches = math.ceil(len(task_vocab)/float(ex_batch_size))

            with torch.no_grad():
                for cur_batch in trange(ex_max_batches, unit="batches", leave=False):

                    cur_vocabs = task_vocab[cur_batch*ex_batch_size:cur_batch*ex_batch_size+ex_batch_size]
                    
                    token_input = en_fr_tokenizer(cur_vocabs, return_tensors="pt", padding=True).to('cuda')
                    translated  = en_fr_model.generate(**token_input)
                    trans_fr    = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

                    token_input = fr_en_tokenizer(trans_fr, return_tensors="pt", padding=True).to('cuda')
                    translated  = fr_en_model.generate(**token_input)
                    trans_en    = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

                    trans_en = [sample.lower() for sample in trans_en]
                    bt_vocab.extend(trans_en)

                    #print(len(bt_vocab))
                    #if len(bt_vocab) > 1000:
                    #    break

                    with open(bt_vocab_path, 'w') as f:
                        for vocab in bt_vocab:
                            f.write(vocab + '\n')

        assert len(bt_vocab) == len(bt_vocab), "Number of original sentences and back-translated ones must match"



        # double verify with synonym dataset (only keep the overlapping ones)
        synonym_pairs = []
        with open('./datasets/word_sim/synonym.txt') as f:
            for line in f:
                line.strip().split()
                synonym_pairs.append(line.strip().split())

        pos_pairs = []
        for i in range(len(bt_vocab)):
            if task_vocab[i] == bt_vocab[i]:
                continue
            elif [task_vocab[i],bt_vocab[i]] in synonym_pairs:
                pos_pairs.append([task_vocab[i],bt_vocab[i]])
            elif [bt_vocab[i],task_vocab[i]] in synonym_pairs:
                pos_pairs.append([bt_vocab[i],task_vocab[i]])

        for item in pos_pairs:
            if item not in self.pos_pairs:
                self.pos_pairs.append(item)
            if [item[1],item[0]] not in self.pos_pairs:
                self.pos_pairs.append([item[1],item[0]])

        logging.info('{} positive pairs collected after back-translation'.format(len(self.pos_pairs)))



    def build_basic_vocab(self):
        '''
            build basic vocabulary from positive pairs
        '''

        for item in self.pos_pairs:
            if item[0] not in self.vocab: self.vocab.append(item[0])
            if item[1] not in self.vocab: self.vocab.append(item[1])

        logging.info("{} Background Vocab Collected from Similar Word Pairs".format(len(self.vocab)))


    def aug_ws_vocab(self):
        '''
            augment vocab through word similarity datasets
        '''

        for _, cur_dataset in self.ws_data.items():
            for pair in cur_dataset:
                w1, w2, _ = pair
                w1 = w1.lower()
                w2 = w2.lower()
                if w1 not in self.vocab:
                    self.vocab.append(w1)
                if w2 not in self.vocab:
                    self.vocab.append(w2)

        logging.info("{} Background Vocab Collected after Adding Word Similarity Vocabs".format(len(self.vocab)))


    def aug_task_vocab(self):
        '''
            augment vocab by task vocabs
        '''

        task_vocab    = []
        if self.task_name == 'SST2':
            with open('./datasets/sent_classification/SST/binary/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])
        elif self.task_name == 'SST5':
            with open('./datasets/sent_classification/SST/fine/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])
        elif self.task_name == 'SICKEntailment':
            with open('./datasets/sent_classification/SICK/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])
        else:
            with open('./datasets/sent_classification/'+self.task_name+'/vocab.txt', 'r') as f:
                for line in f:
                    task_vocab.append(line.split()[0])

        for word in task_vocab:
            if word not in self.vocab:
                self.vocab.append(word)

        logging.info("{} Background Vocab Collected after Adding Taks-Specific Vocabs".format(len(self.vocab)))


    def aug_wiki_vocab(self):
        '''
            augment voacb with wiki vocabulary
        '''

        wiki_vocab = []

        with open('./datasets/word_sim/enwiki_vocab_min200.txt', 'r') as f:
            for line in f: wiki_vocab.append(line.strip().split()[0])

        wiki_vocab = wiki_vocab[:20000]

        for word in wiki_vocab:
            if word not in self.vocab:
                self.vocab.append(word)

        logging.info("{} Background Vocab Collected after Adding Wiki Vocabs".format(len(self.vocab)))

