from base64 import encode
import pandas as pd 
from typing import Union
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from itertools import combinations
import numpy as np

class RankingDataset:

    # Per extension load functions
    __per_ext_load_func = dict(
        csv = pd.read_csv
        )

    __CONFIG = dict(
        ohe_cols = ["property", "scale", "class"],
        text_cols = ["long_common_name", "component"]
    )
    
    
    def __init__(self, 
            datapath:Union[str, Path], 
            separator=",", 
            sentence_transformer='pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',
            datapath_queries:Union[str,Path] = None,
            datapath_query_doc:Union[str,Path] = None
        ):
        self.__data = self.__load_data(Path(datapath), separator)
        self.embedder = sentence_transformer
        self.__data_enc = self.__encode_docs()
        self.__queries = self.__load_data(Path(datapath_queries), separator) if datapath_queries else None
        self.__queries_enc = self.__encode_queries() if datapath_queries else None
        self.__query_doc = self.__load_data(Path(datapath_query_doc), separator) if datapath_query_doc else None
        self.__query_doc_enc = self.__encode_query_doc() if datapath_queries and datapath_query_doc else None


    @property
    def embedder(self):
        return self.__embedder

    @embedder.setter
    def embedder(self, checkpoint_name:str)->None:
        self.__embedder = SentenceTransformer(checkpoint_name)


    def __load_data(self, datapath, separator) -> None:
        fext = datapath.name.split(".")[-1]
        if fext not in self.__per_ext_load_func:
            raise ValueError(f'Invalid extension for file "{datapath.name}" ({fext}). Try again with one of {list(self.__per_ext_load_func.keys())}')
        
        self.__datapath = datapath

        return self.__per_ext_load_func[fext](self.__datapath, sep=separator)

    def __encode_docs(self) -> pd.DataFrame:
        """One hot encode categorical features and transform into the embedding space textual features."""

        data_ohe = self.__data.copy()

        for ohe_col in self.__CONFIG["ohe_cols"]:
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


        for text_col in self.__CONFIG["text_cols"]:
            encoded = self.embedder.encode(data_ohe[text_col])
            data_ohe[text_col] = pd.Series([enc for enc in encoded])
        
        return data_ohe

    def __encode_queries(self) -> pd.DataFrame:
        """Embede the queries"""
        data_copy = self.__queries.copy()

        encoded = self.embedder.encode(data_copy['query'])
        data_copy['query'] = pd.Series([enc for enc in encoded])

        return data_copy

    def __encode_query_doc(self) -> pd.DataFrame:
        """Join the datasets and compute properties"""
        merge1 = pd.merge(self.__queries_enc,self.__query_doc, on="id_query")
        df = pd.merge(merge1, self.__data_enc, on="loinc_num")

        df['cos_sim_query_long_common_name'] = df.apply(lambda row: float(util.cos_sim(row['query'],row['long_common_name'])), axis=1)
        df['cos_sim_query_component'] = df.apply(lambda row: float(util.cos_sim(row['query'],row['component'])), axis=1)

        df = df.drop(columns=['long_common_name', 'component', 'query'])

        return df
    
    def prepare_training_data_pairwise(self, query = None):
        # TODO: Si la query no es None que prepare el dataset con esa query (para predict)
        X, y = [], []
        df_copy = self.__query_doc_enc.drop(columns=['loinc_num'])

        for q in df_copy['id_query'].unique():
            sub_df = df_copy[df_copy['id_query']==q].drop(columns=['id_query'])
            for d1, d2 in combinations(range(sub_df.shape[0]), 2):
                df_diff = sub_df.iloc[d1] - sub_df.iloc[d2]
                X.append(df_diff.drop(columns=['rank']).to_numpy())
                y.append(np.sign(df_diff['rank']))

        return np.array(X), np.array(y)


data = RankingDataset('./data/docs.csv', datapath_queries='./data/queries.csv', datapath_query_doc='./data/query_doc.csv')
pairwise_data_X, pairwise_data_y = data.prepare_training_data_pairwise()
model = LinearSVC(random_state=0, tol=1e-5)
model.fit(pairwise_data_X, pairwise_data_y)
print(model.coef_)
