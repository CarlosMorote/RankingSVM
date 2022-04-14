# Pairwise ranking via SVM

### Lorenzo Alfaro, David

### Morote García, Carlos

#### ETSIINF, UPM

---

</br>

## Introduction

TODO

---

</br>

## Run the code

First of all you have to set the information files in the `data` folder. This data will be used to train the model and consists of three files: `docs.csv`, `queries.csv` and `query_doc.csv` (The name does not have to be explicitly this one). 

 - `docs.csv` is the collection of the documents and must have the following characteristics: `loinc_num`, `long_common_name`, `component`, `system`, `property`, `scale`, `class`, `discourage`, `rank_loinc`.
 - `queries.csv` is the collection of queries by which the model will learn. It consists of `id_query`, `query`.
 - `query_doc.csv` is the relationship between the documents and the queries. It consists of `id_query`, `loinc_num`, `rank`.

This being the only requirement. To execute the code use the following command.

    python pairwise_ranking_retrieval.py *doc_path* *queries_path* *query_doc_path*

</br>


## Folder sctructure

    .
    ├── README.md
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

TODO
