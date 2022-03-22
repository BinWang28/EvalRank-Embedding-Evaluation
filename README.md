## Word & Sentence Embedding Evaluation

**In this project, we provide an easy-to-use toolkit for both word and sentence embedding evaluations.**

<p align="center">
  <img src="img/logo.png" width="600" height="auto" />
</p>

### Update
- **Mar. 22, 2022**
    - More example [scripts](scripts/) for how to test on each supported model.
- **Mar.21, 2022**
    - You can now follow the template to test on your [own embedding model](#Models-and-Quick-Start). 
    - Support a series of sentence embedding models including [InferSent, SimCSE, Sentence_BERT, BERT-Whitening, BERT-Flow, etc.](src/models/sent_emb/).
    - Sentence Embedding Evaluation part is updated.
- **Mar.20, 2022**
    - Word Embedding Evaluation part is updated.

## Outline

<div align="center">

| Section | Description |
|-|-|
| [References](#References)											| References                    	|
| [Evluation Tasks](#Evluation-Tasks)          						| Evluation Tasks                   |
| [Environment Setup](#Environment-Setup) 							| Environments             		    |
| [Models and Quick Start](#Models-and-Quick-Start) 				| Models and Quick Start            |
| [Benchmarking - Word](#Benchmarking---Word)						| Leaderboard              		    |
| [Benchmarking - Sentence](#Benchmarking---Sentence)				| Leaderboard              		    |
| [Acknowledge](#Acknowledge)										| Acknowledge		   		 		|

</div>

### References

If you find our package useful, please cite our paper.
- [**Just Rank: Rethinking Evaluation with Word and Sentence Similarities**](https://arxiv.org/abs/2203.02679)
    - To appear in **ACL 2022 Main Conference**

```
@article{evalrank_2022,
  title={Just Rank: Rethinking Evaluation with Word and Sentence Similarities},
  author={Wang, Bin and Kuo, C.-C. Jay and Li, Haizhou},
  journal={arXiv preprint arXiv:2203.02679},
  year={2022}
}
```
    - Will update the ACL proceeding once it is out.

## Evluation Tasks

The following are the supported evaluation tasks:

- **Word Embedding Evaluation**
    - [EvalRank (Word-Level)](https://arxiv.org/abs/2203.02679)
    - [Word Similarity Tasks](https://arxiv.org/abs/1901.09785)
        - [WS-353](https://dl.acm.org/doi/10.1145/503104.503110), [WS-353-SIM](https://aclanthology.org/N09-1003/), [WS-353-REL](https://aclanthology.org/N09-1003/), [MC-30](https://www.tandfonline.com/doi/abs/10.1080/01690969108406936), [RG-65](https://dl.acm.org/doi/10.1145/365628.365657), [Rare-Word](https://aclanthology.org/W13-3512/), [MEN](https://www.jair.org/index.php/jair/article/view/10857), [MTurk-287](https://dl.acm.org/doi/10.1145/1963405.1963455), [MTurk-771](https://dl.acm.org/doi/10.1145/2339530.2339751), [YP-130](https://arxiv.org/abs/cs/0212033), [SimLex-999](https://aclanthology.org/J15-4004/), [Verb-143](https://aclanthology.org/D14-1034/), [SimVerb-3500](https://aclanthology.org/D16-1235/)
- **Sentence Embedding Evaluation**
    - [EvalRank (Sentence-Level)](https://arxiv.org/abs/2203.02679)
    - Downstream Tasks
        - [MR](https://aclanthology.org/P05-1015/), [CR](https://dl.acm.org/doi/10.1145/1014052.1014073), [SUBJ](https://aclanthology.org/P04-1035/), [MPQA](https://link.springer.com/article/10.1007/s10579-005-7880-9), [SST2](https://aclanthology.org/D13-1170/), [SST5](https://aclanthology.org/D13-1170/), [TREC](https://aclanthology.org/C02-1150/), [MRPC](https://aclanthology.org/C04-1051/), [SICK-E](https://aclanthology.org/L14-1314/), [SCICITE](https://arxiv.org/abs/1904.01608)
    - Semantic Textual Similarity (STS) Tasks
        - STS 12~16, STS-Benchmark, [STR](https://arxiv.org/pdf/2110.04845.pdf)

## Environment Setup

Tested with the following dependencies:
- python==3.8.12
- pytorch==1.11.0
- transformers==4.11.3
- scikit-learn==0.23.2

Please lookinto the details of the following script file for setting up the environment.

    bash environment.sh

## Models and Quick Start

We have supoorted a list of word & sentence embedding models for quick evaluation and benchmarking.

- **Word Embedding Models**
    - Any word embedding files follow this [format](./src/models/word_emb/).
    - Integrate one post-processing method.

- **Word Level Similarity and EvalRank**
    - To test on your own model, simply change the word embedding path.

    ```
    bash word_evaluate.sh

    # To evaluate on your own word embedding model
    update file: word_evaluate.sh
    WORD_EMB_PATH='PATH/TO/WORD/EMBEDDING'
    ```   

- **Sentence Embedding Models**
    - Bag-of-word (averaging word embedding)
    - Bag-of-word with post-processing
    - InferSent
    - BERT
    - BERT-Whitening
    - BERT-Flow
    - Sentence-BERT
    - SimCSE

- **Sentence Level Similarity and EvalRank**
    - You can also easily test your own sentence embedding model using our provided [template](src/models/sent_emb/_customize.py).

    ```
    bash sentence_evaluate.sh

    # To evaluate on your own sentence embedding model modify the following to files
    update file: sentence_evaluate.sh
    SENT_EMB_MODEL='customize'
    update file: ./src/models/sent_emb/_customize.py
    ```

    For better **classification** performance, edit the following part (in file [src/s_evaluation.py](src/s_evaluation.py)):

    ```
    params_senteval = {'task_path': './data/', 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                    'tenacity': 3, 'epoch_size': 2}
    ```
    to
    ```
    params_senteval.update({'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10})
    params_senteval['classifier'] = {'nhid': 50, 'optim': 'adam', 'batch_size': 64,
                                    'tenacity': 5, 'epoch_size': 4}
    ```

**For a complete set of model performance, refer to the bash and log files in [scripts/](scripts/). Simply run the corresponding script for results.**

## Benchmarking - Word

<div align="center">

| Word Embedding (cos) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 3.18 | 1.18 | 3.54 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 13.15 | 4.66 | 15.72 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 12.88 | 4.57 | 14.35 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | **17.22** | **5.77** | **19.99** |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 12.71 | 4.04 | 13.04 |

</div>

- More benchmarking results can be found in this page: [word_evalrank](./benchmarking/word_evalrank.md), [word_similarity](./benchmarking/word_similarity.md).
- More benchmarking results can also be found in scripts and their corresponding logs.


## Benchmarking - Sentence

<div align="center">

| Sentence Embedding (cos) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 41.15 | 28.79 | 49.65 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 61.00 | 44.94 | 74.66 |
| InferSentv1 | 60.72 | 41.92 | 77.21 |
| InferSentv2 | 63.89 | 45.59 | 80.47 |
| BERT(first-last-avg) | 68.01 | 51.70 | 81.91 |
| BERT-whitening | 66.58 | 46.54 | 84.22 |
| Sentence-BERT | 64.12 | 47.07 | 79.05 |
| SimCSE | **69.50** | **52.34** | **84.43** |

</div>

## Acknowledge

- We borrow a big portion of sentence embedding evaluation from [SentEval](https://github.com/facebookresearch/SentEval). Please consider cite their work if you found that part useful.


Contact Info: [bwang28c@gmail.com](mailto:bwang28c@gmail.com).