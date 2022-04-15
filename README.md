# Pairwise ranking via SVM

### Lorenzo Alfaro, David

### Morote García, Carlos

#### ETSIINF, UPM

---

</br>

## Introduction

In this work we describe a basic implementation of a retrieval function via linear support-
vector machines, following the framework proposed in (Joachims, 2002), which lies within the
pairwise ranking paradigm. To that end, we utilize a small sample of data from the LOINC
database to extract a meaningful set of document descriptors both at training and inference
time.

<sub>Joachims, T. (2002). Optimizing search engines using clickthrough data. In Proceedings of
the eighth acm sigkdd international conference on knowledge discovery and data mining
(pp. 133–142).</sub>

---

</br>

## Run the code

First of all you have to set the information files in the `data` folder. This data will be used to train the model and consists of three files: `docs.csv`, `queries.csv` and `query_doc.csv` (The name does not have to be explicitly this one). 

 - `docs.csv` is the collection of the documents and must have the following characteristics: `loinc_num`, `long_common_name`, `component`, `system`, `property`, `scale`, `class`, `discourage`, `rank_loinc`.
 - `queries.csv` is the collection of queries by which the model will learn. It consists of `id_query`, `query`.
 - `query_doc.csv` is the relationship between the documents and the queries. It consists of `id_query`, `loinc_num`, `rank`.

The main script corresponds to `pairwise_ranking_retrieval.py`. This script initializes the engine with the provided parameters and waits to a user's query in order to compute a proper ordering for that query. The script takes as parameters:
 - **datapath_documents**: Filepath to the file containing the collection of documents.
 - **datapath_queries**: Filepath to the file containing the collection of queries.
 - **datapath_query_doc**: Filepath to the file containing the per-query-per-document ranks.
 - **--sentence_transformer**: Name of a custom sentence transformer checkpoint. _Default_: "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
  - **--seed**:Seed to use to grant reproducibility of the training process of the SVM model._Type_: int.  _Default_: 0
  - **--separator**: Data field separator, e.g., ',' in csv files. _Default_: ","
 - **-s --similarity_measure**: Default similarity score function. _Values_: "dot", "cos", "euclidean". _Default_: "cos"
 - **-t --topk**: Top k documents of the rank to show in CLI. _Type_:int . _Default_: 10

To execute the code you can use the following command.

    python pairwise_ranking_retrieval.py *doc_path* *queries_path* *query_doc_path* [optional params]*

</br>


## Folder sctructure

    .
    ├── README.md
    ├── requirements.txt
    ├── data
    │   ├── docs.csv
    │   ├── queries.csv
    │   └── query_doc.csv
    ├── dataset.py
    ├── SVM.py
    └── pairwise_ranking_retrieval.py

---

</br>

## Code dependences

- `numpy` (1.20.3)
- `packaging` (21.0)
- `pandas` (1.3.4)
- `scikit_learn` (1.0.2)
- `sentence_transformers` (2.2.0)
- `torch` (1.11.0)
