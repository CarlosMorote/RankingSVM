import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

class RankSVM(LinearSVC):

    def __init__(self, data, seed = 0):
        print('Initializing RankSVM')
        self.__data = data
        super().__init__(random_state=seed, tol=1e-5) 
        self.fit()
        print('RankSVM initialized')

    def fit(self, sample_weight=None):
        _data = self.__data.prepare_training_data_pairwise()
        super(RankSVM, self).fit(_data[0], _data[1], sample_weight)
        return self

    def rank(self, query: str):
        return self.__rank(self.__data.prepare_training_data_pairwise(query)[0])

    def __rank(self, X):
        return np.argsort(X @ self.coef_.ravel())

    def show_ranking(self, query:str, top=5):
        order = self.rank(query)

        df_copy = self.__data.get_data().copy()

        df_copy['order'] = pd.Series(order)
        df_copy = df_copy.sort_values('order',ascending=True).drop(columns=['order'])

        print(df_copy.head(top))

        return df_copy
