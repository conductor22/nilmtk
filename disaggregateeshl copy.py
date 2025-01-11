from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk.metrics import f1_score, rms_error_power
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd


dataset = DataSet("E:/Users/Megapoort/eshldaten/csv/eshl.h5")
train = DataSet("E:/Users/Megapoort/eshldaten/csv/eshl.h5")
test = DataSet("E:/Users/Megapoort/eshldaten/csv/eshl.h5")

# tz naive error mit set_window() :(
# dataset.store.window = TimeFrame(start="2024-08-01", end="2024-09-01")
# train.store.window = TimeFrame(start="2024-08-01", end="2024-08-16")
# test.store.window = TimeFrame(start="2024-08-16", end="2024-09-01")
# device 1 komplett fucked???
dataset.set_window(start="2024-08-03", end="2024-08-05")
train.set_window(start="2024-08-03", end="2024-08-04")
test.set_window(start="2024-08-04", end="2024-08-05")

print(dataset.buildings[1].elec.mains().power_series_all_data().to_frame().tail())
draw_plot(dataset.buildings[1].elec.mains(), "whole set")
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
draw_plot(train_test_mains, "train and test set")

# train_elec = train.buildings[1].elec.submeters().select_top_k(k=13)
train_elec = train.buildings[1].elec.submeters().select_top_k(k=5)
draw_plot(train_elec, "all meters")

fhmm = FHMM()
fhmm.train(train_elec)
fhmm_output = HDFDataStore("E:/Users/Megapoort/eshldaten/csv/fhmm.h5", "w")

print("*** Training Done ***")

fhmm.disaggregate(test.buildings[1].elec.mains(), fhmm_output)
fhmm_output.close()

print("*** Disaggregation Done ***")

ground_truth = test.buildings[1].elec
fhmm_dataset = DataSet("E:/Users/Megapoort/eshldaten/csv/fhmm.h5")
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
