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

# dataset = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
# train = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
# test = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")

# dataset.set_window(start="2024-08-01", end="2024-09-01")
# train.set_window(start="2024-08-01", end="2024-08-16")
# test.set_window(start="2024-08-16", end="2024-09-01")

building = dataset.buildings[1]  # Replace 1 with the building number

# for appliance in train.buildings[1].elec.appliances:
#     print(appliance)

# dataset.buildings[1].elec.draw_wiring_graph()

# Training plots
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
draw_plot(train_test_mains, "Trainset & Testset Mains")

train_elec = train.buildings[1].elec.submeters()
all_meters = [train.buildings[1].elec.mains(), train.buildings[1].elec.submeters()]
draw_plot(all_meters, "main & submeters")

train_main = [train.buildings[1].elec.mains().power_series_all_data()]  # list of dataframes

train_df = train.buildings[1].elec.mains().power_series_all_data().to_frame() # power_series_all_data() -> series.Series  ,   to_frame() -> frame.DataFrame
train_main = [train_df]
test_df = test.buildings[1].elec.mains().power_series_all_data().to_frame()
test_main = [test_df]



# find out top k meters
top_10 = train.buildings[1].elec.submeters().select_top_k(k=10) # Metergroup

top_10_list = []    # list of dataframes
for meter in top_10.meters:
    df = meter.power_series_all_data().to_frame()
    top_10_list.append(df)

# get the indices
top_10_instances = [meter.instance() for meter in top_10.meters]

train_appliances = []
for i in top_10_instances:
    appliance = train.buildings[1].elec[i]
    appliance_name = appliance.appliances[0].type['type']
    appliance_df = appliance.power_series_all_data().to_frame()
    appliance_data = [appliance_df]

    existing_names = [name for name, _ in train_appliances]
    if appliance_name in existing_names:
        count = sum(1 for name in existing_names if name.startswith(appliance_name))
        appliance_name = f"{appliance_name}_{count + 1}"
    train_appliances.append((appliance_name, appliance_data))

print(train_appliances)

# FHMM disaggregation
fhmm = FHMMExact({})    # 1 n Elemente als Input -> n Elemente als Output
fhmm.partial_fit(train_main=train_main, train_appliances=train_appliances)
fhmm_prediction_list = fhmm.disaggregate_chunk(test_main)   # list of dataframes (nur ein Eintrag)
draw_plot(fhmm_prediction_list)

# CO disaggregation
co = CO({})
co.partial_fit(train_main=train_main, train_appliances=train_appliances)
co_prediction_list = co.disaggregate_chunk(test_main)
draw_plot(co_prediction_list)

# Mean disaggregation
mean = Mean({})
mean.partial_fit(train_main=train_main, train_appliances=train_appliances)
mean_prediction_list = mean.disaggregate_chunk(test_main)
draw_plot(mean_prediction_list)

# Auswertung
# ground_truth = test.buildings[1].elec
# fhmm_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5")
# co_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5")
# # mle_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/mle.h5")
# fhmm_predictions = fhmm_dataset.buildings[1].elec
# co_predictions = co_dataset.buildings[1].elec
# mle_predictions = mle_dataset.buildings[1].elec

# print("FHMM f1-score:")
# print(f1_score(ground_truth=ground_truth, predictions=fhmm_predictions))
# print("CO f1-score:")
# print(f1_score(ground_truth=ground_truth, predictions=co_predictions))
# print("MLE f1-score:")
# print(f1_score(ground_truth=ground_truth, predictions=mle_predictions))
# print("FHMM RMSE:")
# print("CO RMSE:")

# draw_plot(fhmm_predictions, "FHMM Meters")
# draw_plot(co_predictions, "CO Meters")
# draw_plot(mle_predictions, "Mean Meters")
# draw_plot(ground_truth, "Ground Truth")

# meter_info = [
#     {"index": meter.identifier.instance, "type": meter.appliances[0].type if meter.appliances else "unkown"}
#     for meter in top_5_train_elec.meters
# ]
# indices = [meter.identifier.instance for meter in top_5_train_elec.meters]


test_dataframe_list = []
for meter in test.buildings[1].elec.submeters().meters:
    df = meter.power_series_all_data().to_frame()
    test_dataframe_list.append(df)

print(test_dataframe_list)
print(top_10_instances)
print(train.buildings[1].elec)
print(fhmm_prediction_list)

print(top_10_instances)
for i in top_10_instances:
    print(i)

for fhmm, gt in zip(fhmm_prediction_list[0], top_10_instances):
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    fhmm_df.index = pd.date_range(start='2013-01-02 00:00:00', end='2013-01-02 23:59:00', freq='1T', tz='Europe/Berlin')
    print(type(fhmm_df))
    print(fhmm)
    print(gt)
    print(type(test_dataframe_list[gt]))
    df_list = [fhmm_df, test_dataframe_list[gt]]
    draw_plot(fhmm_df)
    draw_plot(test_dataframe_list[gt])
    draw_plot(df_list)

fehler

for column in fhmm_prediction_list[0]:
    print(type(column))
    print(column)
    print(type(fhmm_prediction_list[0][column]))
    df = fhmm_prediction_list[0][column].to_frame()
    draw_plot(df)

for i in top_10_instances:
    # print("index: ", i)
    # print("fhmm: ", fhmm_prediction_list)
    # print("gt: ", test_dataframe_list)
    gt_submeter = test_dataframe_list[i]
    # fhmm_submeter = fhmm_prediction_list[i]
    # co_submeter = co_prediction_list[i]
    # mean_submeter = mean_prediction_list[i]
    draw_plot(gt_submeter)
    
    # plot_this = [gt_submeter, fhmm_submeter, co_submeter, mean_submeter]
    # draw_plot(plot_this, title="index")

fehler
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
