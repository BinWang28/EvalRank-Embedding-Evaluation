## Cosine Similarity

| Word Embedding (cos) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 3.18 | 1.18 | 3.54 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 13.15 | 4.66 | 15.72 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 12.88 | 4.57 | 14.35 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 17.22 | 5.77 | 19.99 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 12.71 | 4.04 | 13.04 |

## L2 Similarity

| Word Embedding (l2) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 37.59 | 35.47 | 37.78 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 11.21 | 3.23 | 13.89 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 9.23 | 3.10 | 12.01 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 14.21 | 4.26 | 17.79 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 5.40 | 1.60 | 7.04 |

## Cosine Similarity with post-processing

| Word Embedding (cos+post-processing) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 3.10 | 1.07 | 3.50 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 13.55 | 5.11 | 15.40 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 13.12 | 4.97 | 14.56 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 17.42 | 6.18 | 19.53 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 12.68 | 3.95 | 13.18 |

## L2 Similarity with post-processing

| Word Embedding (l2+post-processing) | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 37.73 | 35.55 | 37.99 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 11.04 | 3.26 | 13.58 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 9.07 | 2.81 | 12.21 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 12.98 | 3.74 | 17.54 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 5.08 | 1.54 | 6.86 |


**In the previous version of our paper, Table 5 (a) used the wrong version without augmentation with wiki vocabulary. The result is shown as below:**

| Word Embedding (cos) w/o wiki vocab | EvalRank (MRR) | Hits1 | Hits3 |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 4.64 | 2.19 | 5.04 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 18.72 | 8.45 | 22.71 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 19.30 | 8.54 | 22.92 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 25.36 | 11.35 | 31.25 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 21.28 | 9.59 | 24.92 |