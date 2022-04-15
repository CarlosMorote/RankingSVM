import pandas as pd 
from typing import Callable, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations
import numpy as np
import torch

class RankingDataset:

    # Per extension load functions
    __per_ext_load_func = dict(
        csv = pd.read_csv
        )

    __SCORE_FUNC_MAP = dict(
        dot = util.dot_score,
        cos = util.cos_sim,
        euclidean = lambda x, y: torch.tensor([1/(1+torch.nn.functional.pairwise_distance(x, k)) for k in y])
    )

    __similarity_measure = None

    
    __config = dict(
        ohe_cols = ["property", "scale", "class"],
        text_cols = ["long_common_name", "component"],
        doc_pk = "loinc_num",
        query_pk = "id_query"
    )
    
    
    def __init__(self, 
            datapath:Union[str, Path],
            datapath_queries:Union[str,Path],
            datapath_query_doc:Union[str,Path],
            separator=",", 
            sentence_transformer='pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',
            similarity_measure='cos'
        ):
        print('Initializing RankingDataset')
        
        self.similarity_measure = similarity_measure
        self.embedder = sentence_transformer

        self.__data = self.__load_data(Path(datapath), separator)
        self.__data_enc = self.__encode_docs()

        self.__queries = self.__load_data(Path(datapath_queries), separator)
        self.__queries_enc = self.__encode_queries()

        # Prepare training dataset
        self.__query_doc = self.__load_data(Path(datapath_query_doc), separator)
        self.__query_doc_enc = self.__encode_query_doc()

        print('Ranking Dataset initialized')


    @property
    def embedder(self):
        return self.__embedder

    @embedder.setter
    def embedder(self, checkpoint_name:str)->None:
        self.__embedder = SentenceTransformer(checkpoint_name)

    @property
    def data(self):
        return self.__data

    @property
    def train_data(self):
        if not hasattr(self, '__train_data'):
            self.train_data = self.__get_train_data()
        return self.__train_data

    @train_data.setter
    def train_data(self, data):
        self.__train_data  = data

    @property
    def similarity_measure(self) -> Callable:
        return self.__similarity_measure
    
    @similarity_measure.setter
    def similarity_measure(self, similarity_measure: str) -> None:
        self.__similarity_measure = self.__load_similarity_measure(similarity_measure)
        
    def __load_data(self, datapath, separator) -> None:
        fext = datapath.name.split(".")[-1]
        if fext not in self.__per_ext_load_func:
            raise ValueError(f'Invalid extension for file "{datapath.name}" ({fext}). Try again with one of {list(self.__per_ext_load_func.keys())}')
        
        return self.__per_ext_load_func[fext](datapath, sep=separator)


    def __load_similarity_measure(self, similarity_measure: str)->bool:
        if similarity_measure and similarity_measure not in self.__SCORE_FUNC_MAP:
            raise ValueError(f'Invalid compare method. Try again with one of {list(self.__SCORE_FUNC_MAP.keys())}.')
        elif similarity_measure:
            return self.__SCORE_FUNC_MAP[similarity_measure]
        else:
            # Explicit None return clause.
            return None

    def __encode_docs(self) -> pd.DataFrame:
        """One hot encode categorical features and transform into the embedding space textual features.

        Returns
        -------
        pd.DataFrame
            Dataframe with long_common_name and component encoded. Also, the rest of the categorical variables are on-hot encoded.
        """

        data_ohe = self.__data.copy()

        for ohe_col in self.__config["ohe_cols"]:
            data_ohe = data_ohe.join(pd.get_dummies(data_ohe.pop(ohe_col), prefix=ohe_col))

        # Fixed transformations
        
        # discourage
        data_ohe.discourage = data_ohe.discourage.map({'no':0, 'yes':1})

        # system
        mlb = MultiLabelBinarizer()
        data_ohe.system = data_ohe.system.str.split('/')
        data_ohe = data_ohe.join(pd.DataFrame(mlb.fit_transform(data_ohe.pop('system')),
                        columns=[f"system_{i}" for i in mlb.classes_],
                        index=data_ohe.index))


        for text_col in self.__config["text_cols"]:
            encoded = self.embedder.encode(data_ohe[text_col])
            data_ohe[text_col] = pd.Series([enc for enc in encoded])
        
        return data_ohe

    def __encode_queries(self) -> pd.DataFrame:
        """Encode the queries with the embedding

        Returns
        -------
        pd.DataFrame
            Dataframe with the query property encoded
        """
        data_copy = self.__queries.copy()

        encoded = self.embedder.encode(data_copy['query'])
        data_copy['query'] = pd.Series([enc for enc in encoded])

        return data_copy

    def __encode_query_doc(self) -> pd.DataFrame:
        """Join the datasets and compute properties such as the cos similarity between
        long_common_name and query; component and query too.

        Returns
        -------
        pd.DataFrame
            Dataframe with all the information of queries and its related document.
            The cos similarity its also added
        """

        # Performs join between the data
        merge = pd.merge(self.__queries_enc,self.__query_doc, on=self.__config["query_pk"])
        df = pd.merge(merge, self.__data_enc, on=self.__config["doc_pk"])

        # Compute the cos similarity against the query vs long_common_name and component
        for text_col in self.__config["text_cols"]:
            df[f'cos_sim_{text_col}'] = df.apply(lambda row: float(util.cos_sim(row['query'],row[text_col])), axis=1)

        # Drop those columns that the model cannot process
        df = df.drop(columns=self.__config["text_cols"] + ["query"])

        return df


    def __get_train_data(self):
        X, y = [], []
        # Training
        df_copy = self.__query_doc_enc.drop(columns=self.__config["doc_pk"])

        # Compute per each different query
        query_pk = self.__config["query_pk"]
        for q in df_copy[query_pk].unique():
            sub_df = df_copy[df_copy[query_pk]==q].drop(columns=[query_pk])
            # Compute all the pairwise instances (combinations)
            for d1, d2 in combinations(range(sub_df.shape[0]), 2):
                diff = sub_df.iloc[d1] - sub_df.iloc[d2]
                X.append(diff.drop(labels=['rank']).to_numpy())
                y.append(np.sign(diff['rank']))
        return np.array(X), np.array(y)

    def get_test_data(self, query):
        # Prediction
        query_embedded = self.embedder.encode(query)

        # Compute the cos similiarities and drop columns
        df_copy = self.__data_enc.copy()
        for text_col in self.__config["text_cols"]:
            df_copy[f'cos_sim_{text_col}'] = df_copy.apply(lambda row: float(self.similarity_measure(query_embedded,row[text_col])), axis=1)
        df_copy = df_copy.drop(columns=self.__config["text_cols"] + [self.__config["doc_pk"]])

        return df_copy.to_numpy()



