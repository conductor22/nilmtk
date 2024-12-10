from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk.metrics import f1_score, rms_error_power
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd
'''
import h5py

with h5py.File("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5", "r") as f:
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
'''


dataset = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
train = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
test = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
dataset.buildings[1].elec.draw_wiring_graph()
plt.show()
# print(dataset.metadata.get('timezone'))
# timeframe = dataset.buildings[1].elec.mains().get_timeframe()
# print("Start:", timeframe.start, "| tzinfo:", timeframe.start.tzinfo)
# print("End:", timeframe.end, "| tzinfo:", timeframe.end.tzinfo)

# tz naive error mit set_window() :(
dataset.store.window = TimeFrame(start="2024-08-01", end="2024-09-01")
train.store.window = TimeFrame(start="2024-08-01", end="2024-08-16")
test.store.window = TimeFrame(start="2024-08-16", end="2024-09-01")

# scheiß panda version
# print(next(dataset.buildings[1].elec.mains().load()))

draw_plot(dataset.buildings[1].elec.mains(), "whole set")
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
draw_plot(train_test_mains, "train and test set")

train_elec = train.buildings[1].elec.submeters().select_top_k(k=20)
draw_plot(train_elec, "all meters")

fhmm = FHMM()
fhmm.train(train_elec)
fhmm_output = HDFDataStore("E:/Users/Megapoort/eshldaten/oneetotwelve/fhmm.h5", "w")

print("*** Training Done ***")

fhmm.disaggregate(test.buildings[1].elec.mains(), fhmm_output)
fhmm_output.close()

print("*** Disaggregation Done ***")

ground_truth = test.buildings[1].elec
fhmm_dataset = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/fhmm.h5")
fhmm_predictions = fhmm_dataset.buildings[1].elec

draw_plot(ground_truth, "Ground Truth")
draw_plot(fhmm_predictions, "FHMM Meters")


meter_info = [
    {"index": meter.identifier.instance, "type": meter.appliances[0].type if meter.appliances else "unkown"}
    for meter in train_elec.meters
]
indices = [meter.identifier.instance for meter in train_elec.meters]

top_20_in_test = []
for i, index in enumerate(indices):
    meter = ground_truth[index]
    top_20_in_test.append(meter)

print(top_20_in_test)

for i, index in enumerate(indices):
    device = ground_truth[index]
    fhmm_device_predictions = fhmm_predictions[index]
    all_meters = [device, fhmm_device_predictions]
    title = "Device " + str(index)
    draw_plot(all_meters, title)
