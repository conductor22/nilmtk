from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk.metrics import f1_score, rms_error_power
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd
#import h5py


# with h5py.File('C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5', 'a') as f:
#     if '/METER' not in f:
#         group = f.create_group('/METER')
#     else:
#         group = f['/METER']

#     group.attrs['model'] = 'Generic Meter Clamp'
#     group.attrs['sample_period'] = 1
#     group.attrs['measurements'] = [
#         {'physical_quantity': 'power', 'type': 'active', 'unit': 'W'},
#         {'physical_quantity': 'power', 'type': 'reactive', 'unit': 'var'}
#     ]





# with h5py.File("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5", "r") as f:
#     def print_attrs(name, obj):
#         print(f"Node: {name}")
#         for key, val in obj.attrs.items():
#             print(f"  {key}: {val}")

#     f.visititems(print_attrs)

import h5py

with h5py.File("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5", "r") as f:
    print("Top-level keys:", list(f.keys()))  # building1
    
    building_group = f["building1"]
    
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




dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5")
'''
for building in dataset.buildings.values():
    print(f"building: {building}")
    for meter in building.elec.meters:
        print(f"meter: {meter}")

building = dataset.buildings[1]  # Replace with your building ID
for meter in building.elec.meters:
    print(f"Meter {meter}:")
    df = next(meter.load())
    print(df.head())  # Print first few rows to see the data
    print(df.shape)   # Print the shape to see if data exists
'''
#draw_plot(dataset.buildings[1].elec, "test")




import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_hdf("C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5", key="building1/elec/meter1/table", columns=['P1'])

print(df.head())

df = pd.DataFrame(df['data'].tolist(), columns=["time", "P1", "P2", "P3", "Q1", "Q2", "Q3"])
pprint(df)
df['time'] = pd.to_datetime(df['time'], unit='ns')

print(df.head())

plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['P1'])
plt.title("Power Consumption Over Time (P1)")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()