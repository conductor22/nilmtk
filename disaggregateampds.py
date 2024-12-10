from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
# from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, MLE
from nilmtk.disaggregate import CO, FHMMExact, Mean
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

print(dir(train.buildings[1].elec.mains()))

# dataset.buildings[1].elec.draw_wiring_graph()

# Training plots
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
draw_plot(train_test_mains, "Trainset & Testset Mains")

print(train.buildings[1].elec)
print("***")
train_elec = train.buildings[1].elec.submeters()
print("Train elecs: ", train_elec)

all_meters = [train.buildings[1].elec.mains(), train.buildings[1].elec.submeters()]
draw_plot(all_meters, "main & top 5 meters")
print("###")
train_main = [train.buildings[1].elec.mains().power_series_all_data()]
print("Type of train_main:", type(train_main))
print("Type of train_main:", type(train.buildings[1].elec.mains().power_series_all_data()))
df = train.buildings[1].elec.mains().power_series_all_data().to_frame()
print("Now a Dataframe?", type(df))
train_main = [df]
train_appliances = []

test_df = test.buildings[1].elec.mains().power_series_all_data().to_frame()
test_main = [test_df]

# for meter in train_elec.meters:
#     for appliance in meter.appliances:
#         appliance_name = appliance.type['type']
#         print(appliance_name)

for i in range(2, 22):
    appliance_name = train.buildings[1].elec[i].appliances[0].type['type']
    appliance_df = train.buildings[1].elec[i].power_series_all_data().to_frame()
    appliance_data = [appliance_df]
    train_appliances.append((appliance_name, appliance_data))

# print(train_appliances)


print("df type: ", type(df))
print("train_main type: ", type(train_main))
print("type of content of train_main: ", type(train_main[0]))

# FHMM disaggregation
fhmm = FHMMExact({})
fhmm.partial_fit(train_main=train_main, train_appliances=train_appliances)
# fhmm_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5", "w")

fhmm_prediction_list = fhmm.disaggregate_chunk(test_main)
print(fhmm_prediction_list)
print(type(fhmm_prediction_list))

plt.figure(figsize=(12, 8))
for i, chunk in enumerate(fhmm_prediction_list):
    for appliance in chunk.columns:
        plt.plot(chunk.index, chunk[appliance], label=appliance)
plt.xlabel('Time')
plt.ylabel('Power Consumption (W)')
plt.title('Disaggregated Appliance Power')
plt.legend()
plt.show()

# fhmm_output.close()

# CO disaggregation
co = CO({})
co.partial_fit(train_main=train_main, train_appliances=train_appliances)
# co_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5", "w")

co_prediction_list = co.disaggregate_chunk(test_main)
# co_output.close()

# Mean disaggregation
# mle = MLE()
# mle.train(top_5_train_elec)
# mle_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/mle.h5", "w")

# mle.disaggregate(test.buildings[1].elec.mains(), co_output)
# mle_output.close()

# Auswertung
ground_truth = test.buildings[1].elec
fhmm_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5")
co_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5")
# mle_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/mle.h5")
fhmm_predictions = fhmm_dataset.buildings[1].elec
co_predictions = co_dataset.buildings[1].elec
# mle_predictions = mle_dataset.buildings[1].elec

print("FHMM f1-score:")
print(f1_score(ground_truth=ground_truth, predictions=fhmm_predictions))
print("CO f1-score:")
print(f1_score(ground_truth=ground_truth, predictions=co_predictions))
# print("MLE f1-score:")
# print(f1_score(ground_truth=ground_truth, predictions=mle_predictions))
# print("FHMM RMSE:")
# print("CO RMSE:")

draw_plot(fhmm_predictions, "FHMM Meters")
draw_plot(co_predictions, "CO Meters")
# draw_plot(mle_predictions, "Mean Meters")
draw_plot(ground_truth, "Ground Truth")

# meter_info = [
#     {"index": meter.identifier.instance, "type": meter.appliances[0].type if meter.appliances else "unkown"}
#     for meter in top_5_train_elec.meters
# ]
# indices = [meter.identifier.instance for meter in top_5_train_elec.meters]

top_5_in_test = []
for i, index in enumerate(indices):
    meter = ground_truth[index]
    top_5_in_test.append(meter)
draw_plot(top_5_in_test, "Top 5 train elecs in test dataset")

for i, index in enumerate(indices):
    device = ground_truth[index]
    fhmm_device_predictions = fhmm_predictions[index]
    co_device_predictions = co_predictions[index]
    # mle_device_predictions = mle_predictions[index]
    all_meters = [device, fhmm_device_predictions, co_device_predictions]   #   , mle_device_predictions
    title = "Device " + str(index)
    draw_plot(all_meters, title)
