import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

class RankSVM(LinearSVC):
    """_summary_

    Parameters
    ----------
    LinearSVC : _type_
        _description_
    """
    def __init__(self, data, seed = 0):
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_
        seed : int, optional
            _description_, by default 0
        """
        print('Initializing RankSVM')
        self.dataset = data
        super().__init__(random_state=seed, tol=1e-5) 
        self.fit()
        print('RankSVM initialized')

    def fit(self, sample_weight=None):
        """_summary_

        Parameters
        ----------
        sample_weight : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        super(RankSVM, self).fit(*self.dataset.train_data, sample_weight)
        return self

    def rank(self, query: str):
        return np.argsort(self.dataset.get_test_data(query) @ self.coef_.ravel())

    def show_ranking(self, query:str, top=5):
        order = self.rank(query)

        df_copy = self.dataset.data.copy()

        df_copy['order'] = pd.Series(order)
        df_copy = df_copy.sort_values('order',ascending=True).drop(columns=['order'])

        print(df_copy.head(top))

        return df_copy
