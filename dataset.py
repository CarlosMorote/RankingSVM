from base64 import encode
import pandas as pd 
from typing import Union
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer

class RankingDataset:

    # Per extension load functions
    __per_ext_load_func = dict(
        csv = pd.read_csv
        )

    __CONFIG = dict(
        ohe_cols = ["property", "scale", "class"],
        text_cols = ["long_common_name", "component"]
    )
    
    
    def __init__(self, datapath:Union[str, Path], separator=";", sentence_transformer='pritamdeka/S-Biomed-Roberta-snli-multinli-stsb'):
        self.__data = self.__load_data(Path(datapath), separator)
        self.embedder = sentence_transformer
        self.__data_enc = self.__encode_data()


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

    def __encode_data(self) -> None:
        """One hot encode categorical features and transform into the embedding space textual features."""

        data_ohe = self.__data.copy()

        for ohe_col in self.__CONFIG["ohe_cols"]:
            data_ohe = data_ohe.join(pd.get_dummies(data_ohe.pop(ohe_col), prefix=ohe_col))

        # Fixed transformations
        
        # discourage
        data_ohe.discourage = data_ohe.discourage.map({'no':False, 'yes':True})

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
    
    def prepare_training_data(self, queries):
        #TODO: aquí hay cosas que ver bien cuando tengas terminado toda la parte de preprocesamiento.
        # La idea es que cuando se llame a esta función, todas las transformaciones INDEPENDIENTES de las
        # queries estén ya hechas (es lo q se hace en el método self.__encode_data()). Una vez aquí, 
        # hay que transformar query a embedding space (self.embedder.encode) y calcular métricas de 
        # similaridad con respecto de cada elemento en las columnas en self.__CONFIG["text_cols"].
        # El problema es ver cómo gestionamos de una manera no súper chapucera el hecho de generar "a la vez", un súper dataframe
        # con tamaño N*q donde N es el tamaño del dataset y q el número de queries, sino generarlo bajo demanda, p.ej., devolver un
        # iterador que te devuelva N datos por cada llamada (las features para UNA query en concreto a cada momento).
        # Aquí es donde también tenemos que gestionar el tema del cruce de datos (el cross product entre query y doc
        # teniendo en cuenta su relación, que viene data por el RANKING manual que se decida)
        # Igualmente, esto lo vemos mañana cuando esté todo el resto si quieres

        pass


aux = RankingDataset('./data/docs.csv')