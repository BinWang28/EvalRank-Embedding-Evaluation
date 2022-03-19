# Word & Sentence Embedding Evaluation

<p align="center">
  <img src="img/logo.png" width="600" height="auto" />
</p>

## Outline

<div align="center">

| Section | Description |
|-|-|
| [Reference](#Reference)												| Reference                    		|
| [Introduction](#Introduction)          							| Overview                 		    |
| [Environment Setup](#Environment-Setup) 							| Environments             		    |
| [Supported Architectures](#Supported-Architectures) 				| Example Architectures    		    |
| [Quick User Guide](#Quick-User-Guide)							    | Scripts                 		    |
| [Benchmarking - Word](#Benchmarking---Word)						| Leaderboard              		    |
| [Benchmarking - Sentence](#Benchmarking---Sentence)				| Leaderboard              		    |
| [Acknowledge](#Acknowledge)										| Acknowledge		   		 		|

</div>

## Reference

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

## Introduction

In this project, we provide an easy-to-use toolkit for both word and sentence embedding evaluations.

- **Word Embedding Evaluation**
    - [EvalRank (Word-Level)](https://arxiv.org/abs/2203.02679)
    - [Word Similarity Tasks](https://arxiv.org/abs/1901.09785)
        - [WS-353](https://dl.acm.org/doi/10.1145/503104.503110), [WS-353-SIM](https://aclanthology.org/N09-1003/), [WS-353-REL](https://aclanthology.org/N09-1003/), [MC-30](https://www.tandfonline.com/doi/abs/10.1080/01690969108406936), [RG-65](https://dl.acm.org/doi/10.1145/365628.365657), [Rare-Word](https://aclanthology.org/W13-3512/), [MEN](https://www.jair.org/index.php/jair/article/view/10857), [MTurk-287](https://dl.acm.org/doi/10.1145/1963405.1963455), [MTurk-771](https://dl.acm.org/doi/10.1145/2339530.2339751), [YP-130](https://arxiv.org/abs/cs/0212033), [SimLex-999](https://aclanthology.org/J15-4004/), [Verb-143](https://aclanthology.org/D14-1034/), [SimVerb-3500](https://aclanthology.org/D16-1235/)
- **Sentence Embedding Evaluation**
    - [EvalRank (Sentence-Level)](https://arxiv.org/abs/2203.02679)
    - Downstream Tasks
        - [MR](https://aclanthology.org/P05-1015/), [CR](https://dl.acm.org/doi/10.1145/1014052.1014073), [SUBJ](https://aclanthology.org/P04-1035/), [MPQA](https://link.springer.com/article/10.1007/s10579-005-7880-9), [SST2](https://aclanthology.org/D13-1170/), [SST5](https://aclanthology.org/D13-1170/), [TREC](https://aclanthology.org/C02-1150/), [MRPC](https://aclanthology.org/C04-1051/), [SICK-E](https://aclanthology.org/L14-1314/), [SCICITE](https://arxiv.org/abs/1904.01608)
    - Semantic Textual Similarity (STS) Tasks
        - STS 12 ~ 16, STS-Benchmark, [STR](https://arxiv.org/pdf/2110.04845.pdf)

## Environment Setup

Please lookinto the details of the following script file for setting up the environment.

    bash environment.sh

## Supported Architectures 

We have supoorted a list of word & sentence embedding models for quick evaluation and benchmarking.

- Word Embeddings
    - Any word embedding files follow the [THIS FORMAT](./models/word_emb/).
    - Integrate one post-processing method.
- Sentence Embeddings
    - TODO: BERT, SBERT, BERT-flow, BERT-whitening, xxx

## Quick User Guide

- **Word Level Similarity and EvalRank**

```
bash word_evaluate.sh
```

## Benchmarking - Word


| Word Embedding (cos) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 13.15 | 4.66 | 15.72 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 12.88 | 4.57 | 14.35 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 17.22 | 5.77 | 19.99 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | xx | --- | --- |


- More benchmarking results can be found in this page: [word_evalrank](./benchmarking/word_evalrank.md), [word_similarity](./benchmarking/word_similarity.md).


## Benchmarking - Sentence

TODO: table for benchmarking results


## Acknowledge

    xx xx xx xx


Contact Info: [bwang28c@gmail.com](mailto:bwang28c@gmail.com).


To be finished... (I will try to finished it before April)

