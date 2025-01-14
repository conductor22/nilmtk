from collections import OrderedDict
import pandas as pd
from sklearn.cluster import KMeans
from nilmtk.disaggregate import Disaggregator

class KMeansDisaggregator(Disaggregator):

    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = 'KMeans'
        self.num_clusters = params.get('num_clusters', 10)
        self.app_names = []

    def partial_fit(self, train_main, **load_kwargs):
        """
        Train using KMeans clustering.
        """
        print(".........................KMeans partial_fit.................")

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.dropna()

        X = train_main.values.reshape((-1, 1))

        if not len(X):
            print(f"No samples, skipping...")
            return

        assert X.ndim == 2
        self.X = X

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        kmeans.fit(X)
        self.model = kmeans
        print(f"Learnt model")

        print("KMeans partial_fit end.................")

    def disaggregate_chunk(self, test_mains_list):
        """
        Disaggregate the test data according to the model learnt previously.
        """
        test_prediction_list = []

        for test_mains in test_mains_list:
            if len(test_mains) == 0:
                tmp = pd.DataFrame(index=test_mains.index, columns=[f'Appliance_{i+1}' for i in range(self.num_clusters)])
                test_prediction_list.append(tmp)
            else:
                length = len(test_mains.index)
                temp = test_mains.values.reshape(length, 1)

                labels = self.model.predict(temp)

                appliance_powers = pd.DataFrame(index=test_mains.index, columns=[f'Appliance_{i+1}' for i in range(self.num_clusters)])
                
                for cluster in range(self.num_clusters):
                    appliance_powers[f'Appliance_{cluster+1}'] = (labels == cluster).astype(int) * test_mains.values.flatten()

                test_prediction_list.append(appliance_powers)

        return test_prediction_list