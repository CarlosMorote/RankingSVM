import pandas as pd 
from typing import Union
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations
import numpy as np

from SVM import RankSVM

class RankingDataset:

    # Per extension load functions
    __per_ext_load_func = dict(
        csv = pd.read_csv
        )

    __config = dict(
        ohe_cols = ["property", "scale", "class"],
        text_cols = ["long_common_name", "component"],
        doc_pk = "loinc_num",
        query_pk = "id_query"
    )
    
    
    def __init__(self, 
            datapath:Union[str, Path], 
            separator=",", 
            sentence_transformer='pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',
            datapath_queries:Union[str,Path] = None,
            datapath_query_doc:Union[str,Path] = None
        ):
        print('Initializing RankingDataset')
        self.__data = self.__load_data(Path(datapath), separator)
        self.embedder = sentence_transformer
        self.__data_enc = self.__encode_docs()
        self.__queries = self.__load_data(Path(datapath_queries), separator) if datapath_queries else None
        self.__queries_enc = self.__encode_queries() if datapath_queries else None
        self.__query_doc = self.__load_data(Path(datapath_query_doc), separator) if datapath_query_doc else None
        self.__query_doc_enc = self.__encode_query_doc() if datapath_queries and datapath_query_doc else None
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
        
    def __load_data(self, datapath, separator) -> None:
        fext = datapath.name.split(".")[-1]
        if fext not in self.__per_ext_load_func:
            raise ValueError(f'Invalid extension for file "{datapath.name}" ({fext}). Try again with one of {list(self.__per_ext_load_func.keys())}')
        
        self.__datapath = datapath

        return self.__per_ext_load_func[fext](self.__datapath, sep=separator)

    def __encode_docs(self) -> pd.DataFrame:
        """One hot encode categorical features and transform into the embedding space textual features."""

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
        """Encode the queries with the embedding"""
        data_copy = self.__queries.copy()

        encoded = self.embedder.encode(data_copy['query'])
        data_copy['query'] = pd.Series([enc for enc in encoded])

        return data_copy

    def __encode_query_doc(self) -> pd.DataFrame:
        """Join the datasets and compute properties"""

        # Performs join between the data
        merge = pd.merge(self.__queries_enc,self.__query_doc, on=self.__config["query_pk"])
        df = pd.merge(merge, self.__data_enc, on=self.__config["doc_pk"])

        # Compute the cos similarity against the query vs long_common_name and component
        for text_col in self.__config["text_cols"]:
            df[f'cos_sim_{text_col}'] = df.apply(lambda row: float(util.cos_sim(row['query'],row[text_col])), axis=1)

        # Drop those columns that the model cannot process
        df = df.drop(columns=self.__config["text_cols"] + ["query"])

        return df
    
    @property
    def train_data(self):
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
        if query:
            # Prediction
            query_embedded = self.embedder.encode(query)

            # Compute the cos similiarities and drop columns
            df_copy = self.__data_enc.copy()
            for text_col in self.__config["text_cols"]:
                df_copy[f'cos_sim_{text_col}'] = df_copy.apply(lambda row: float(util.cos_sim(query_embedded,row[text_col])), axis=1)
            df_copy = df_copy.drop(columns=self.__config["text_cols"] + [self.__config["doc_pk"]])

            return df_copy.to_numpy()


if __name__=='__main__':
    data = RankingDataset('./data/docs.csv', datapath_queries='./data/queries.csv', datapath_query_doc='./data/query_doc.csv')
    rankingModel = RankSVM(data)
    while (query := input("Query (type q to quit): ").strip().lower()) != 'q':
        rankingModel.show_ranking(query, top=10)

#pairwise_data_X, pairwise_data_y = data.prepare_training_data_pairwise()
#model = LinearSVC(random_state=0, tol=1e-5)
#model.fit(pairwise_data_X, pairwise_data_y)
#print(model.coef_)
#test_X, _ = data.prepare_training_data_pairwise("Blood")
#print(np.argsort(test_X @ model.coef_.ravel()))
