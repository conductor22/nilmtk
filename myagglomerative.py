from collections import OrderedDict
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from nilmtk.disaggregate import Disaggregator

class AgglomerativeClusteringDisaggregator(Disaggregator):

    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = 'AgglomerativeClustering'
        self.num_clusters = params.get('num_clusters', 10)
        self.app_names = []

    def partial_fit(self, train_main, **load_kwargs):
        """
        Train using Agglomerative Clustering.
        """
        print(".........................AgglomerativeClustering partial_fit.................")

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.dropna()

        X = train_main.values.reshape((-1, 1))

        if not len(X):
            print(f"No samples, skipping...")
            return

        assert X.ndim == 2
        self.X = X

        agglomerative = AgglomerativeClustering(n_clusters=self.num_clusters)
        agglomerative.fit(X)
        self.model = agglomerative
        print(f"Learnt model")

        print("AgglomerativeClustering partial_fit end.................")

    def disaggregate_chunk(self, test_mains_list):
        """
        Disaggregate the test data according to the model learnt previously.
        """
        test_prediction_list = []

        for test_mains in test_mains_list:
            test_mains = test_mains.dropna()
            X_test = test_mains.values.reshape((-1, 1))

            if not len(X_test):
                print(f"No samples in test data, skipping...")
                continue

            assert X_test.ndim == 2

            predictions = self.model.fit_predict(X_test)
            test_prediction_list.append(predictions)

        return test_prediction_list