from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk.metrics import f1_score, rms_error_power
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd


# Datensets erstellen
dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")
dataset_elec = dataset.buildings[1].elec

train = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")
test = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")

dataset.set_window(start="2013-01-01", end="2013-01-03")
train.set_window(start="2013-01-01", end="2013-01-02")
test.set_window(start="2013-01-02", end="2013-01-03")

# Training plots
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
draw_plot(train_test_mains, "Trainset & Testset Mains")

top_5_train_elec = train.buildings[1].elec.submeters().select_top_k(k=5)
all_meters = [train.buildings[1].elec.mains(), top_5_train_elec]
draw_plot(all_meters, "main & top 5 meters")


# FHMM disaggregation
fhmm = FHMM()
fhmm.train(top_5_train_elec)
fhmm_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5", "w")

fhmm.disaggregate(test.buildings[1].elec.mains(), fhmm_output)
fhmm_output.close()

# CO disaggregation
co = CombinatorialOptimisation()
co.train(top_5_train_elec)
co_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5", "w")

co.disaggregate(test.buildings[1].elec.mains(), co_output)
co_output.close()


# Auswertung
ground_truth = test.buildings[1].elec
fhmm_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5")
co_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5")
fhmm_predictions = fhmm_dataset.buildings[1].elec
co_predictions = co_dataset.buildings[1].elec

print("FHMM f1-score:")
print(f1_score(ground_truth=ground_truth, predictions=fhmm_predictions))
print("CO f1-score:")
print(f1_score(ground_truth=ground_truth, predictions=co_predictions))
# print("FHMM RMSE:")
# print("CO RMSE:")

draw_plot(fhmm_predictions, "FHMM Meters")
draw_plot(co_predictions, "CO Meters")
draw_plot(ground_truth, "Ground Truth")

meter_info = [
    {"index": meter.identifier.instance, "type": meter.appliances[0].type if meter.appliances else "unkown"}
    for meter in top_5_train_elec.meters
]
indices = [meter.identifier.instance for meter in top_5_train_elec.meters]

for i, index in enumerate(indices):
    device = ground_truth[index]
    fhmm_device_predictions = fhmm_predictions[index]
    co_device_predictions = co_predictions[index]
    all_meters = [device, fhmm_device_predictions, co_device_predictions]
    title = "Device " + str(index)
    draw_plot(all_meters, title)
