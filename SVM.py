import numpy as np
from sklearn.svm import LinearSVC

class RankSVM(LinearSVC):

    def __init__(self, data):
        print('Initializing RankSVM')
        self.__data = data
        super().__init__() 
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

    def show_ranking(self, order):
        pass