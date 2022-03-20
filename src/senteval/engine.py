# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.trec import TRECEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.str import STREval
from senteval.sst import SSTEval
from senteval.scicite import SCICITEEval
from senteval.probing import *

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['SCICITE', 'CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STR',
                           'Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'CR':
            self.evaluation = CREval(tpath + 'sent_classification/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + 'sent_classification/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + 'sent_classification/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + 'sent_classification/SUBJ', seed=self.params.seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + 'sent_classification/SST/binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + 'sent_classification/SST/fine', nclasses=5, seed=self.params.seed)
        elif name == 'SCICITE':
            self.evaluation = SCICITEEval(tpath + 'sent_classification/SCICITE', nclasses=3, seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + 'sent_classification/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + 'sent_classification/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + 'sent_similarity/SICK', seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + 'sent_similarity/STS/STSBenchmark', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + 'sent_classification/SICK', seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + 'sent_similarity/STS/' + fpath, seed=self.params.seed)
        elif name == 'STR':
            self.evaluation = STREval(tpath + 'sent_similarity/STR/', seed=self.params.seed)

        # Probing Tasks
        elif name == 'Length':
                self.evaluation = LengthEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'WordContent':
                self.evaluation = WordContentEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'Depth':
                self.evaluation = DepthEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'TopConstituents':
                self.evaluation = TopConstituentsEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'BigramShift':
                self.evaluation = BigramShiftEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'Tense':
                self.evaluation = TenseEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'SubjNumber':
                self.evaluation = SubjNumberEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'ObjNumber':
                self.evaluation = ObjNumberEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'OddManOut':
                self.evaluation = OddManOutEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'CoordinationInversion':
                self.evaluation = CoordinationInversionEval(tpath + '/probing', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
