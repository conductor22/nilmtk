from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore, ElecMeter
import matplotlib.pyplot as plt
from pprint import pprint
# from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, MLE
from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
import nilmtk.metrics
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd

def create_df(meter):
    df_active_power = meter.power_series_all_data(ac_type='active').to_frame()
    # df_reactive_power = meter.power_series_all_data(ac_type='reactive').to_frame()
    # df = pd.concat([df_active_power, df_reactive_power], axis=1)
    df = df_active_power
    return df


# Datensets erstellen
dataset = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshlP3.h5")
train = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshlP3.h5")
test = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshlP3.h5")

print(type(dataset.buildings[1].elec.mains()))

start_date = pd.Timestamp("2024-08-02")
end_date = pd.Timestamp("2024-08-04")

ratio = 0.8 # 80% train, 20% test
train_test_split_point = start_date + (end_date - start_date) * ratio

dataset.set_window(start=start_date, end=end_date)
train.set_window(start=start_date, end=train_test_split_point)
test.set_window(start=train_test_split_point, end=end_date)

# dataset.buildings[1].elec.draw_wiring_graph()

# Training plots
train_test_mains = [train.buildings[1].elec.mains(), test.buildings[1].elec.mains()]
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


aggregate = test_elec.meters[0].power_series_all_data(ac_type='active').to_frame()
for meter in test_elec.meters:
    if meter.instance() == "1":
        print("skipped")
        continue
    print(meter.instance())
    df = meter.power_series_all_data(ac_type='active').to_frame()
    df.columns = pd.MultiIndex.from_tuples([("power", "active")])
    aggregate += df
whatever_list = [aggregate, test.buildings[1].elec.mains()]
# whatever_list = [aggregate]
# whatever_list.append(test.buildings[1].elec.mains().power_series_all_data().to_frame())
# draw_plot(whatever_list, "aggregate")



# Main train and test
train_df = train.buildings[1].elec.mains().power_series_all_data().to_frame() # power_series_all_data() -> series.Series  ,   to_frame() -> frame.DataFrame
train_main = [train_df]
test_df = test.buildings[1].elec.mains().power_series_all_data().to_frame()
test_main = [test_df]




# find out top k meters
top_10 = train.buildings[1].elec.submeters().select_top_k(k=9) # Metergroup

top_10_list = []    # list of dataframes
for meter in top_10.meters:
    df = meter.power_series_all_data().to_frame()
    top_10_list.append(df)

# get the indices
top_10_instances = [meter.instance() for meter in top_10.meters]

train_appliances = []
for i in top_10_instances:
    appliance = train.buildings[1].elec[i]
    appliance_name = "unkown"
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
# draw_plot(fhmm_prediction_list)

# CO disaggregation
co = CO({})
co.partial_fit(train_main=train_main, train_appliances=train_appliances)
co_prediction_list = co.disaggregate_chunk(test_main)
# draw_plot(co_prediction_list)

# Mean disaggregation
mean = Mean({})
mean.partial_fit(train_main=train_main, train_appliances=train_appliances)
mean_prediction_list = mean.disaggregate_chunk(test_main)
# draw_plot(mean_prediction_list)

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


import mymetrics

# test main is a df

test_tuple_list = []

gt_dict = {index: test_dataframe_list[index - 1] for index in top_10_instances}

df = fhmm_prediction_list[0]
predictions = [df[[col]] for col in df.columns]
''' predictions in order of highest to lowest power
    gt_dict in order of top_10_instances that rank from highest to lowerst power
'''

print("fhmm error_in_assigned_energy: ", mymetrics.error_in_assigned_energy(predictions, gt_dict))
# print("co error_in_assigned_energy: ", mymetrics.error_in_assigned_energy(co_prediction_list, gt_dict))
# print("mean error_in_assigned_energy: ", mymetrics.error_in_assigned_energy(mean_prediction_list, gt_dict))
print("**********************************")
print("fhmm fraction_energy_assigned_correctly: ", mymetrics.fraction_energy_assigned_correctly(predictions, gt_dict))
# print("co fraction_energy_assigned_correctly: ", mymetrics.fraction_energy_assigned_correctly(co_prediction_list, gt_dict))
# print("mean fraction_energy_assigned_correctly: ", mymetrics.fraction_energy_assigned_correctly(mean_prediction_list, gt_dict))
print("**********************************")
print("fhmm f1_score: ", mymetrics.f1_score(predictions, gt_dict))
# print("co f1_score: ", mymetrics.f1_score(co_prediction_list, test_dataframe_list))
# print("mean f1_score: ", mymetrics.f1_score(mean_prediction_list, test_dataframe_list))
# print("**********************************")
# print("fhmm mean_normalized_error_power: ", mymetrics.mean_normalized_error_power(fhmm_prediction_list, test_dataframe_list))
# print("co mean_normalized_error_power: ", mymetrics.mean_normalized_error_power(co_prediction_list, test_dataframe_list))
# print("mean mean_normalized_error_power: ", mymetrics.mean_normalized_error_power(mean_prediction_list, test_dataframe_list))
# print("**********************************")
# print("fhmm rms_error_power: ", mymetrics.rms_error_power(fhmm_prediction_list, test_dataframe_list))
# print("co rms_error_power: ", mymetrics.rms_error_power(co_prediction_list, test_dataframe_list))
# print("mean rms_error_power: ", mymetrics.rms_error_power(mean_prediction_list, test_dataframe_list))
# print("**********************************")

print("fhmm rmse: ", mymetrics.rms_error_power(predictions, gt_dict))
print("fhmm mae: ", mymetrics.mean_absolute_error_power(predictions, gt_dict))
print("fhmm mne: ", mymetrics.mean_normalized_error_power(predictions, gt_dict))

# list = []
# for fhmm, gt in zip(fhmm_prediction_list[0], top_10_instances):
#     index = gt - 1
#     fhmm_df = fhmm_prediction_list[0][fhmm]
#     gt_df = test_dataframe_list[index]
#     print("**********")
#     error_series = mymetrics.error_in_assigned_energy(fhmm_df, gt_df)
#     error_df = error_series.to_frame()
#     list = [fhmm_df, gt_df, error_df]
#     draw_plot(error_df, title="error df")
#     print("**********")
# fehler

for fhmm, co, mean, gt in zip(fhmm_prediction_list[0], co_prediction_list[0], mean_prediction_list[0], top_10_instances):
    # in order of highest to lowest power (natural order of fhmm_prediction_list and top_10_instances)
    index = gt - 1  # the first index of gt is 0 but i want to compare to the instance number and not the index - the indices go from 2 to 21
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    co_df = co_prediction_list[0][co].to_frame()
    mean_df = mean_prediction_list[0][mean].to_frame()
    # fhmm_df.index = pd.date_range(start='2013-01-02 00:00:00', end='2013-01-02 23:59:00', freq='1T', tz='Europe/Berlin')
    df_list = [fhmm_df, co_df, mean_df, test_dataframe_list[index]]
    # draw_plot(fhmm_df)
    # draw_plot(test_dataframe_list[index])
    # print("fhmm error_in_assigned_energy: ", mymetrics.error_in_assigned_energy(fhmm_df, test_dataframe_list[index]))
    # print("co error_in_assigned_energy: ", mymetrics.error_in_assigned_energy(co_df, test_dataframe_list[index]))
    # print("mean error_in_assigned_energy: ", mymetrics.error_in_assigned_energy(mean_df, test_dataframe_list[index]))
    print(index)
    draw_plot(df_list)

all_prediction_meters = fhmm_prediction_list[0]["unkown"].to_frame().copy()
all_prediction_meters.columns = pd.MultiIndex.from_tuples([("power", "active")])
for i in fhmm_prediction_list[0]:
    if i == "unkown":
        continue
    df = fhmm_prediction_list[0][i].to_frame()
    df.columns = pd.MultiIndex.from_tuples([("power", "active")])
    # all_prediction_meters = all_prediction_meters.add(df, fill_value=0)
    all_prediction_meters += df

list = [all_prediction_meters, test.buildings[1].elec.mains().power_series_all_data().to_frame()]
draw_plot(list, title="aggregate of all meters")
