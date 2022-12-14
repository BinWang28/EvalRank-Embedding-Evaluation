#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals

import os
import io
import json
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier


class SCICITEEval(object):
    def __init__(self, task_path, nclasses=3, seed=1111):
        self.seed = seed

        assert nclasses in [3]
        self.nclasses = nclasses
        logging.info('Transfer task: SciCite classification')

        train = self.loadFile(os.path.join(task_path, 'train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'dev.txt'))
        test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.scicite_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.scicite_data['train']['X'] + self.scicite_data['dev']['X'] + \
                  self.scicite_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        scicite_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            label2id = {'background': 0, 'method': 1, 'result': 2}
            for line in f:
                if self.nclasses == 3:
                    line_data = json.loads(line)
                    scicite_data['y'].append(int(label2id[line_data['label']]))
                    line_data['text'] = "".join([char for char in line_data['text'].lower() if char in 'abcdefghijklmnopqrstuvwxyz '])
                    scicite_data['X'].append(line_data['text'].split())

        assert max(scicite_data['y']) == self.nclasses - 1
        return scicite_data

    def run(self, params, batcher):
        scicite_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.scicite_data:
            logging.debug('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.scicite_data[key]['X'],
                                     self.scicite_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.scicite_data[key]['X'], self.scicite_data[key]['y'] = map(list, zip(*sorted_data))

            scicite_embed[key]['X'] = []
            for ii in range(0, len(self.scicite_data[key]['y']), bsize):
                batch = self.scicite_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                scicite_embed[key]['X'].append(embeddings)
            scicite_embed[key]['X'] = np.vstack(scicite_embed[key]['X'])
            scicite_embed[key]['y'] = np.array(self.scicite_data[key]['y'])
            logging.debug('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': scicite_embed['train']['X'],
                                 'valid': scicite_embed['dev']['X'],
                                 'test': scicite_embed['test']['X']},
                              y={'train': scicite_embed['train']['y'],
                                 'valid': scicite_embed['dev']['y'],
                                 'test': scicite_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            SciCite classification\n'.format(devacc, testacc))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(scicite_embed['dev']['X']),
                'ntest': len(scicite_embed['test']['X'])}
