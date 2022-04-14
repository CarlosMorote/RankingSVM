import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from dataset import RankingDataset

class RankSVM(LinearSVC):
    """
    RankSVM merge a Linear SVC with ranking.
    A model it is fit but the prediction model its override to generate a rank but prediction.
    """

    def __init__(self, data:RankingDataset, seed = 0):
        """Class constructor

        Parameters
        ----------
        data : RankingDataset
            object which contains thw whole data to fit the model and perform ranking afterwards
        seed : int, optional
            To control the random generator, by default 0
        """
        print('Initializing RankSVM')
        self.dataset = data
        super().__init__(random_state=seed, tol=1e-5) 
        self.fit()
        print('RankSVM initialized')

    def fit(self, sample_weight=None) :
        """Fit the data within the LinearSVM to train it

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight
        """
        super(RankSVM, self).fit(*self.dataset.train_data, sample_weight)
        return self

    def rank(self, query: str) -> np.array:
        """Generates a ranking based on a query.

        Parameters
        ----------
        query : str
            query which models the rank

        Returns
        -------
        array-like of shape (n_docs,)
            array where each position correspond to a document in the same index of the data property.
            each value of the array is the position in the ranking
        """
        return np.argsort(self.dataset.get_test_data(query) @ self.coef_.ravel())

    def show_ranking(self, query:str, top=5) -> pd.DataFrame:
        """Show the register ranked after performing the rank itself

        Parameters
        ----------
        query : str
            query which models the rank
        top : int, optional
            max instances to show, by default 5

        Returns
        -------
        Dataframe
            dataframe sorted by ranking
        """
        order = self.rank(query)

        df_copy = self.dataset.data.copy()

        df_copy['order'] = pd.Series(order)
        df_copy = df_copy.sort_values('order',ascending=True).drop(columns=['order'])

        print(df_copy.head(top))

        return df_copy
