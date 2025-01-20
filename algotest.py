from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
# from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, MLE
from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.metrics import f1_score, rms_error_power
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd
import kmeansalgo
import aggloalgo

# import nilmtk_contrib
# print(nilmtk_contrib.__version__)

# Datensets erstellen
dataset = DataSet("C:/Users/ieh-buergin/Desktop/eshl.h5")
train = DataSet("C:/Users/ieh-buergin/Desktop/eshl.h5")
test = DataSet("C:/Users/ieh-buergin/Desktop/eshl.h5")

df = dataset.buildings[1].elec.mains().power_series_all_data().to_frame()

# start_date = df.index[0].date()
# end_date = df.index[-1].date()
# print(start_date)
# print(end_date)

start_date = pd.Timestamp("2024-08-02")
end_date = pd.Timestamp("2024-08-03")

ratio = 0.8 # 80% train, 20% test
train_test_split_point = start_date + (end_date - start_date) * ratio

dataset.set_window(start=start_date, end=end_date)
train.set_window(start=start_date, end=train_test_split_point)
test.set_window(start=train_test_split_point, end=end_date)

# dataset = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
# train = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")
# test = DataSet("E:/Users/Megapoort/eshldaten/oneetotwelve/eshl.h5")

# dataset.set_window(start="2024-08-01", end="2024-09-01")
# train.set_window(start="2024-08-01", end="2024-08-16")
# test.set_window(start="2024-08-16", end="2024-09-01")


# dataset.buildings[1].elec.draw_wiring_graph()

# Training plots
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
draw_plot(train_test_mains, "Trainset & Testset Mains")

train_elec = train.buildings[1].elec.submeters()
all_meters = [train.buildings[1].elec.mains(), train.buildings[1].elec.submeters()]
draw_plot(all_meters, "main & submeters")

# Main train and test
train_df = train.buildings[1].elec.mains().power_series_all_data().to_frame() # power_series_all_data() -> series.Series  ,   to_frame() -> frame.DataFrame
train_main = [train_df]
test_df = test.buildings[1].elec.mains().power_series_all_data().to_frame()
test_main = [test_df]




# find out top k meters
top_10 = train.buildings[1].elec.submeters().select_top_k(k=10) # Metergroup

top_10_instances = []
top_10_list = []    # list of dataframes
for meter in top_10.meters:
    df = meter.power_series_all_data().to_frame()
    top_10_list.append(df)

    instance = meter.instance()
    top_10_instances.append(instance)   # indices

print(top_10_instances)

# get the indices
top_10_instances = [meter.instance() for meter in top_10.meters]

train_appliances = []
for i in top_10_instances:
    appliance = train.buildings[1].elec[i]
    appliance_name = "appliance " + str(i)
    appliance_df = appliance.power_series_all_data().to_frame()
    appliance_data = [appliance_df]

    # existing_names = [name for name, _ in train_appliances]
    # if appliance_name in existing_names:
    #     count = sum(1 for name in existing_names if name.startswith(appliance_name))
    #     appliance_name = f"{appliance_name}_{count + 1}"
    # train_appliances.append((appliance_name, appliance_data))

    train_appliances.append((appliance_name, appliance_data))

print(train_appliances)

# FHMM disaggregation
fhmm = FHMMExact({})    # 1 n Elemente als Input -> n Elemente als Output
fhmm.partial_fit(train_main=train_main, train_appliances=train_appliances)
fhmm_prediction_list = fhmm.disaggregate_chunk(test_main)   # list of dataframes (nur ein Eintrag)
draw_plot(fhmm_prediction_list)

# KMEANS disaggregation
kmeans_params = {'num_clusters': 2}
kmeans = kmeansalgo.KMeansDisaggregator(kmeans_params)
kmeans.partial_fit(train_main)
kmeans_prediction_list = kmeans.disaggregate_chunk(test_main)
draw_plot(kmeans_prediction_list)

# KMEANS disaggregation
kmeans2_params = {'num_clusters': 3}
kmeans2 = kmeansalgo.KMeansDisaggregator(kmeans2_params)
kmeans2.partial_fit(train_main)
kmeans2_prediction_list = kmeans2.disaggregate_chunk(test_main)
draw_plot(kmeans2_prediction_list)

# KMEANS disaggregation
kmeans3_params = {'num_clusters': 8}
kmeans3 = kmeansalgo.KMeansDisaggregator(kmeans3_params)
kmeans3.partial_fit(train_main)
kmeans3_prediction_list = kmeans3.disaggregate_chunk(test_main)
draw_plot(kmeans3_prediction_list)

# KMEANS disaggregation
kmeans4_params = {'num_clusters': 12}
kmeans4 = kmeansalgo.KMeansDisaggregator(kmeans4_params)
kmeans4.partial_fit(train_main)
kmeans4_prediction_list = kmeans4.disaggregate_chunk(test_main)
draw_plot(kmeans4_prediction_list)


# selbst mit 2 schon extrem aufwendig
agglo_params = {'num_clusters': 2}
agglo = aggloalgo.AgglomerativeClusteringDisaggregator(agglo_params)
agglo.partial_fit(train_main)
agglo_prediction_list = agglo.disaggregate_chunk(test_main)
draw_plot(agglo_prediction_list)

# müssen metergroups sein oder so
# f1_fhmm = f1_score(fhmm_prediction_list[0], test_main)
# f1_co = f1_score(co_prediction_list[0], test_main)
# f1_mean = f1_score(mean_prediction_list[0], test_main)

# print(f"F1 FHMM: {f1_fhmm}")
# print(f"F1 CO: {f1_co}")
# print(f"F1 Mean: {f1_mean}")

# create list of all meters in test dataset
test_dataframe_list = []
for meter in test.buildings[1].elec.submeters().meters:
    df = meter.power_series_all_data().to_frame()
    appliance_type = meter.label()
    df.columns = [appliance_type]
    test_dataframe_list.append(df)

for fhmm, kmeans, kmeans2, kmeans3, kmeans4, gt in zip(fhmm_prediction_list[0], kmeans_prediction_list[0], kmeans2_prediction_list[0], kmeans3_prediction_list[0], kmeans4_prediction_list[0], top_10_instances):
    index = gt - 2  # the first index of gt is 0 but i want to compare to the instance number and not the index - the indices go from 2 to 21
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    kmeans_df = kmeans_prediction_list[0][kmeans].to_frame()
    kmeans2_df = kmeans2_prediction_list[0][kmeans2].to_frame()
    kmeans3_df = kmeans3_prediction_list[0][kmeans3].to_frame()
    kmeans4_df = kmeans4_prediction_list[0][kmeans4].to_frame()
    # fhmm_df.index = pd.date_range(start='2013-01-02 00:00:00', end='2013-01-02 23:59:00', freq='1T', tz='Europe/Berlin')
    df_list = [fhmm_df, kmeans_df, kmeans2_df, kmeans3_df, kmeans4_df, test_dataframe_list[index]]
    # draw_plot(fhmm_df)
    # draw_plot(test_dataframe_list[index])
    draw_plot(df_list)


kmeans_prediction_lists = [
    kmeans_prediction_list[0],
    kmeans2_prediction_list[0],
    kmeans3_prediction_list[0],
    kmeans4_prediction_list[0],
]
draw_plot(fhmm_prediction_list)
draw_plot(test_dataframe_list, title="test data all appliances")

for idx, kmeans_prediction in enumerate(kmeans_prediction_lists, start=1):
    draw_plot(kmeans_prediction, title=f"kmeans{idx}")

for idx, kmeans_prediction in enumerate(kmeans_prediction_lists, start=1):
    kmeans_meters = kmeans_prediction["Appliance_1"].to_frame().copy()
    kmeans_meters.columns = pd.MultiIndex.from_tuples([("power", "active")])
    for i in kmeans_prediction:
        if i == "Appliance_1":
            continue
        df = kmeans_prediction[i].to_frame()
        df.columns = pd.MultiIndex.from_tuples([("power", "active")])
        kmeans_meters += df
    newlist = [kmeans_meters, test.buildings[1].elec.mains().power_series_all_data().to_frame()]
    draw_plot(newlist, title=f"aggregate of all meters kmeans{idx}")