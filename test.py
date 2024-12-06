from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk.metrics import f1_score, rms_error_power
from nilmtk.utils import get_datastore
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd
from pprint import pprint
import h5py
# store = pd.HDFStore("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5")
# print(store.info())
# df = store.select('/building1/elec/meter1')
# #pprint(dir(df))

# print(df.columns)


dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5")
print()
print("Buildings:", dataset.buildings.keys())
for building_id, building in dataset.buildings.items():
    print(f"\nBuilding {building_id}:")
    print(building.metadata)

print()

for building_id, building in dataset.buildings.items():
    print(f"\nBuilding {building_id}:")
    print("Available Meters:", building.elec.meters)

print()

for building_id, building in dataset.buildings.items():
    for meter in building.elec.meters:
        print(f"\nBuilding {building_id}, Meter {meter.identifier}:")
        print(meter.metadata)

print()

print("main meters")
print(dataset.buildings[1].elec.mains())

'''
datastore = DataSet("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5")
print("Top-level keys:", list(datastore.keys()))  # building1
    
building_group = datastore["building1"]
    
print("Groups and Datasets within building1:", list(building_group.keys()))
    
elec_group = building_group["elec"]
    
print("Groups and Datasets within elec:", list(elec_group.keys()))
    
for meter_key in elec_group.keys():
    meter_data = elec_group[meter_key]
        
    if isinstance(meter_data, h5py.Dataset):
        print(f"Dataset for {meter_key}:")
        print("Shape:", meter_data.shape)
        print("First 5 rows of data:", meter_data[:5])
    else:
        print(f"{meter_key} is a group")
        print(f"Content of {meter_key}: {list(meter_data.keys())}")

        for dataset_key in meter_data.keys():
            dataset = meter_data[dataset_key]
            if isinstance(dataset, h5py.Dataset):
                print(f"Dataset {dataset_key}:")
                print("Shape:", dataset.shape)
                print("First 5 rows of data:", dataset[:5])
'''
# datastore.buildings[1].elec.mains().plot()
# plt.show()
