from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from nilmtk.disaggregate import Disaggregator

class KMeansDisaggregator(Disaggregator):

    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = 'KMeans'
        self.num_clusters = params.get('num_clusters', 2)
        self.app_names = []

    def partial_fit(self, train_main, **load_kwargs):
        """
        Train using KMeans clustering.
        """
        print(".........................KMeans partial_fit.................")

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.dropna()

        X = train_main.values   # already 2d

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
        # number of clusters just determines the energy levels
        # for test_main in test_mains_list:
        #     print("test_main: ", test_main)
        #     if self.model is None:
        #         raise ValueError("Model has not been trained. Call partial_fit first.")
            
        #     test_main = test_main.dropna()
        #     X_test = test_main.values  # Convert DataFrame to numpy array
            
        #     if not len(X_test):
        #         print("No valid samples for disaggregation.")
        #         return None
            
        #     # Predict cluster labels using the trained model
        #     labels = self.model.predict(X_test)
            
        #     # Assign power values to clusters
        #     cluster_centers = self.model.cluster_centers_
        #     cluster_assignments = np.array([cluster_centers[label] for label in labels])
            
        #     # Return the DataFrame with assigned clusters and estimated power per cluster
        #     test_main["cluster"] = labels
        #     test_main[[('power', 'active'), ('power', 'reactive')]] = cluster_assignments
            
        #     return test_main

        # kinda shit
        # results = []
        
        # for test_main in test_mains_list:
        #     if self.model is None:
        #         raise ValueError("Model has not been trained. Call partial_fit first.")

        #     test_main = test_main.dropna()
        #     X_test = test_main.values  # Convert DataFrame to numpy array
            
        #     if not len(X_test):
        #         print("No valid samples for disaggregation.")
        #         return None
            
        #     # Predict cluster labels using the trained model
        #     labels = self.model.predict(X_test)

        #     # Get cluster centers (each row represents a cluster's power levels)
        #     cluster_centers = self.model.cluster_centers_

        #     # Create DataFrame to store appliance-wise power disaggregation
        #     columns = [(f'Appliance_{i+1}', 'active') for i in range(self.num_clusters)] + \
        #             [(f'Appliance_{i+1}', 'reactive') for i in range(self.num_clusters)]
        #     disaggregated_df = pd.DataFrame(0, index=test_main.index, columns=pd.MultiIndex.from_tuples(columns))

        #     # Assign power values based on cluster labels
        #     for i, label in enumerate(labels):
        #         active_power, reactive_power = cluster_centers[label]  # Extract cluster center power values
        #         disaggregated_df.loc[test_main.index[i], (f'Appliance_{label+1}', 'active')] = active_power
        #         disaggregated_df.loc[test_main.index[i], (f'Appliance_{label+1}', 'reactive')] = reactive_power

        #     results.append(disaggregated_df)
        
        # return results
        ''''''
        # test_prediction_list = []
        # for test_mains in test_mains_list:
        #     if len(test_mains) == 0:
        #         # If no data, create an empty DataFrame for each appliance
        #         tmp = pd.DataFrame(index=test_mains.index, columns=[(f'Appliance_{i+1}', 'active') for i in range(self.num_clusters)] + 
        #                                                 [(f'Appliance_{i+1}', 'reactive') for i in range(self.num_clusters)])
        #         test_prediction_list.append(tmp)
        #     else:
        #         length = len(test_mains.index)
        #         # Extract active and reactive power from the input DataFrame
        #         active_power = test_mains[('power', 'active')].values
        #         reactive_power = test_mains[('power', 'reactive')].values

        #         # Stack both active and reactive power together for clustering
        #         temp = np.stack([active_power, reactive_power], axis=1)

        #         # Predict cluster labels for the test data
        #         labels = self.model.predict(temp)

        #         # Initialize DataFrame for appliance power (both active and reactive)
        #         appliance_powers = pd.DataFrame(index=test_mains.index,
        #                                         columns=[(f'Appliance_{i+1}', 'active') for i in range(self.num_clusters)] +
        #                                                 [(f'Appliance_{i+1}', 'reactive') for i in range(self.num_clusters)])

        #         # Distribute the power among appliances (clusters)
        #         for cluster in range(self.num_clusters):
        #             # Active and reactive power assigned to appliances based on labels
        #             appliance_powers[(f'Appliance_{cluster+1}', 'active')] = (labels == cluster).astype(int) * active_power
        #             appliance_powers[(f'Appliance_{cluster+1}', 'reactive')] = (labels == cluster).astype(int) * reactive_power

        #         # Add the disaggregated powers to the prediction list
        #         test_prediction_list.append(appliance_powers)
        #         print("test_prediction_list: ", test_prediction_list)

        # return test_prediction_list
        ''''''






        test_prediction_list = []

        for test_mains in test_mains_list:
            if len(test_mains) == 0:
                tmp = pd.DataFrame(index=test_mains.index, columns=[f'Appliance_{i+1}' for i in range(self.num_clusters)])
                test_prediction_list.append(tmp)
            else:
                length = len(test_mains.index)
                temp = test_mains.values

                labels = self.model.predict(temp)

                appliance_powers = pd.DataFrame(index=test_mains.index, columns=[f'Appliance_{i+1}' for i in range(self.num_clusters)])
                    
                for cluster in range(self.num_clusters):
                    appliance_powers[f'Appliance_{cluster+1}'] = (labels == cluster).astype(int) * temp.sum(axis=1)

                test_prediction_list.append(appliance_powers)
                print("test_prediction_list: ", test_prediction_list)

        return test_prediction_list
    


