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

def create_df(meter):
    df_active_power = meter.power_series_all_data(ac_type='reactive').to_frame()
    # df_reactive_power = meter.power_series_all_data(ac_type='reactive').to_frame()
    # df = pd.concat([df_active_power, df_reactive_power], axis=1)
    df = df_active_power
    return df

# Datensets erstellen
dataset = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshlQsum.h5")
train = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshlQsum.h5")
test = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshlQsum.h5")

df = create_df(dataset.buildings[1].elec.mains())

start_date = pd.Timestamp("2024-08-02")
end_date = pd.Timestamp("2024-08-30")
end_date = pd.Timestamp("2024-08-30")

ratio = 0.8 # 80% train, 20% test
train_test_split_point = start_date + (end_date - start_date) * ratio

dataset.set_window(start=start_date, end=end_date)
train.set_window(start=start_date, end=train_test_split_point)
test.set_window(start=train_test_split_point, end=end_date)

dataset_elecs = dataset.buildings[1].elec.submeters()
# dataset.buildings[1].elec.draw_wiring_graph()

# Training plots
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
# draw_plot(train_test_mains, "Trainset & Testset Mains")
# draw_plot(train_test_mains, "Trainset & Testset Mains")

train_elec = train.buildings[1].elec.submeters()
train_list = []
for meter in train_elec.meters:
    # df = meter.power_series_all_data().to_frame()
    # test_list.append(df)
    df = create_df(meter)
    train_list.append(df)
train_list.append(train.buildings[1].elec.mains())
# draw_plot(train_list, "main & submeters")

test_elec = test.buildings[1].elec.submeters()
# aggregate = train.buildings[1].elec.submeters().power_series_all_data().to_frame()

test_list = []
for meter in test_elec.meters:
    # df = meter.power_series_all_data().to_frame()
    # test_list.append(df)
    df = create_df(meter)
    test_list.append(df)

test_list.append(create_df(test.buildings[1].elec.mains()))
# draw_plot(test_list, "test main & submeters")


# aggregate = test_elec.meters[0].power_series_all_data(ac_type='active').to_frame()
aggregate = create_df(dataset_elecs.meters[0])
for meter in dataset_elecs.meters:
    print("meter instance:", meter.instance())
    if meter.instance() == 1:
        print("skipped")
        continue
    df = create_df(meter)
    df.columns = pd.MultiIndex.from_tuples([("power", "reactive")])
    aggregate += df

dataset_main_df = create_df(dataset.buildings[1].elec.mains())
# whatever_list = [aggregate, test.buildings[1].elec.mains()]
whatever_list = [dataset_main_df, aggregate]
# whatever_list = [aggregate]
# whatever_list.append(test.buildings[1].elec.mains().power_series_all_data().to_frame())
draw_plot(whatever_list, "aggregate")

difference = dataset_main_df - aggregate
# total_difference = difference.cumsum()

average_hour = difference.resample('H').mean()
cum_average_hour = average_hour.cumsum()

absolute_difference = difference.abs()
cum_sum = absolute_difference.sum().sum()
print(cum_sum)
aaa = [difference, absolute_difference, cum_sum]
draw_plot(aaa)

new_list = [dataset_main_df, aggregate, difference, average_hour, cum_average_hour]
draw_plot(new_list, title="difference")


fehler
# Main train and test
train_active = train.buildings[1].elec.mains().power_series_all_data(ac_type='active').to_frame()
train_reactive = train.buildings[1].elec.mains().power_series_all_data(ac_type='reactive').to_frame()
train_df = pd.concat([train_active, train_reactive], axis=1)

test_active = test.buildings[1].elec.mains().power_series_all_data(ac_type='active').to_frame()
test_reactive = test.buildings[1].elec.mains().power_series_all_data(ac_type='reactive').to_frame()
test_df = pd.concat([test_active, test_reactive], axis=1)

def create_df(meter):
    df_active_power = meter.power_series_all_data(ac_type='active').to_frame()
    df_reactive_power = meter.power_series_all_data(ac_type='reactive').to_frame()
    df = pd.concat([df_active_power, df_reactive_power], axis=1)
    return df


# find out top k meters
top_10 = train.buildings[1].elec.submeters().select_top_k(k=10) # Metergroup

top_10_list = []    # list of dataframes
for meter in top_10.meters:
    df = meter.power_series_all_data().to_frame()
    top_10_list.append(df)

# get the indices
top_10_instances = [meter.instance() for meter in top_10.meters]

train_appliances = []
df_list = []
for i in top_10_instances:
    appliance = train.buildings[1].elec[i]
    appliance_name = "unkown"
    appliance_df = create_df(appliance)
    appliance_data = [appliance_df]
    df_list.append(appliance_df)

    existing_names = [name for name, _ in train_appliances]
    if appliance_name in existing_names:
        count = sum(1 for name in existing_names if name.startswith(appliance_name))
        appliance_name = f"{appliance_name}_{count + 1}"
        
    train_appliances.append((appliance_name, appliance_data))



draw_plot(df_list)

# FHMM disaggregation
params = {'num_of_states': 2}
fhmm = FHMMExact(params)


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
'''
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
'''

for fhmm in fhmm_prediction_list[0]:
    copy_list = fhmm_prediction_list[0]
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    copy_list.append(fhmm_df)
    # draw_plot(fhmm_df)
    # draw_plot(test_dataframe_list[index])
    draw_plot(copy_list)


all_prediction_meters = fhmm_prediction_list[0]["unkown"].to_frame().copy()
all_prediction_meters.columns = pd.MultiIndex.from_tuples([("power", "active")])
for i in fhmm_prediction_list[0]:
    print("i: ", i)
    if i == "unkown":
        continue
    df = fhmm_prediction_list[0][i].to_frame()
    df.columns = pd.MultiIndex.from_tuples([("power", "active")])
    # all_prediction_meters = all_prediction_meters.add(df, fill_value=0)
    all_prediction_meters += df

list = [all_prediction_meters, test.buildings[1].elec.mains().power_series_all_data().to_frame()]
draw_plot(list, title="aggregate of all meters")
