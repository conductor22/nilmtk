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
import nilmtk_contrib

print(nilmtk_contrib.__version__)

# Datensets erstellen
dataset = DataSet("E:/Users/Megapoort/nilmdata/ampds/AMPds2.h5")
train = DataSet("E:/Users/Megapoort/nilmdata/ampds/AMPds2.h5")
test = DataSet("E:/Users/Megapoort/nilmdata/ampds/AMPds2.h5")

df = dataset.buildings[1].elec.mains().power_series_all_data().to_frame()

# start_date = df.index[0].date()
# end_date = df.index[-1].date()
# print(start_date)
# print(end_date)

start_date = pd.Timestamp("2013-01-01")
end_date = pd.Timestamp("2013-02-01")

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

for fhmm, co, mean, gt in zip(fhmm_prediction_list[0], co_prediction_list[0], mean_prediction_list[0], top_10_instances):
    index = gt - 2  # the first index of gt is 0 but i want to compare to the instance number and not the index - the indices go from 2 to 21
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    co_df = co_prediction_list[0][co].to_frame()
    mean_df = mean_prediction_list[0][mean].to_frame()
    # fhmm_df.index = pd.date_range(start='2013-01-02 00:00:00', end='2013-01-02 23:59:00', freq='1T', tz='Europe/Berlin')
    df_list = [fhmm_df, co_df, mean_df, test_dataframe_list[index]]
    # draw_plot(fhmm_df)
    # draw_plot(test_dataframe_list[index])
    draw_plot(df_list)


print(type(fhmm_prediction_list[0]))
print()
print(type(fhmm_prediction_list[0]["heat pump"]))

all_meters = fhmm_prediction_list[0].copy()
draw_plot(all_meters)
for fhmm in fhmm_prediction_list[0]:
    df = fhmm_prediction_list[0][fhmm].to_frame()
    # all_meters = all_meters.add(df, fill_value=0)
    all_meters += df

draw_plot(all_meters, title="aggregate of all meters")