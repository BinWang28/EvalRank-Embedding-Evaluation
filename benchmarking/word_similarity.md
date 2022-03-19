## Cosine Similarity

| Word Embedding (cos) | WS-353-ALL (Pearson) | WS-353-ALL (Spearman) | WS-353-ALL (Kendall) |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 31.86 | 22.20 | 16.04 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 54.81 | 52.03 | 37.91 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 61.66 | 64.24 | 46.54 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 67.71 | 69.22 | 50.95 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 67.99 | 74.42 | 54.75 |


## L2 Similarity

| Word Embedding (l2) | WS-353-ALL (Pearson) | WS-353-ALL (Spearman) | WS-353-ALL (Kendall) |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 7.51 | 24.63 | 17.23 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 27.34 | 46.92 | 33.75 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 31.39 | 39.25 | 27.63 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 31.54 | 48.66 | 34.70 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 28.31 | 43.55 | 30.16 |


## Cosine Similarity with post-processing

| Word Embedding (cos+post-processing) | WS-353-ALL (Pearson) | WS-353-ALL (Spearman) | WS-353-ALL (Kendall) |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 30.97 | 21.28 | 15.41 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 63.73 | 61.65 | 45.24 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 61.74 | 64.71 | 46.75 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 69.62 | 72.71 | 53.69 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 67.99 | 74.80 | 55.19 |

## L2 Similarity with post-processing

| Word Embedding (l2+post-processing) | WS-353-ALL (Pearson) | WS-353-ALL (Spearman) | WS-353-ALL (Kendall) |
| :--- | :---: | :---: | :---: |
| toy_emb.txt | 7.25 | 22.23 | 15.46 |
| [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) | 27.71 | 47.30 | 33.95 |
| [GoogleNews-vectors-negative300.txt](https://code.google.com/archive/p/word2vec/) | 31.29 | 39.17 | 27.50 |
| [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html) | 31.75 | 49.73 | 35.48 |
| [dict2vec-300d.vec](https://github.com/tca19/dict2vec) | 28.20 | 43.31 | 29.97 |