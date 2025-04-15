from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore, ElecMeter
import matplotlib.pyplot as plt
from pprint import pprint
# from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, MLE
from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
import nilmtk.metrics
import nilmtk.utils
from matplotlib import rcParams
from plotting import draw_plot, draw_plot2, draw_stackedplot
import pandas as pd
import mymetrics
import numpy as np

def create_df(meter):
    df_active_power = meter.power_series_all_data(ac_type='active').to_frame()
    # df_reactive_power = meter.power_series_all_data(ac_type='reactive').to_frame()
    # df = pd.concat([df_active_power, df_reactive_power], axis=1)
    df = df_active_power
    return df
 

# Datensets erstellen
dataset = DataSet("C:/Users/ieh-buergin/Desktop/allcsv/eshlP2.h5")
train = DataSet("C:/Users/ieh-buergin/Desktop/allcsv/eshlP2.h5")
test = DataSet("C:/Users/ieh-buergin/Desktop/allcsv/eshlP2.h5")

start_date = pd.Timestamp("2024-08-02")
end_date = pd.Timestamp("2024-08-03")

ratio = 0.8 # 80% train, 20% test
train_test_split_point = start_date + (end_date - start_date) * ratio

dataset.set_window(start=start_date, end=end_date)
train.set_window(start=start_date, end=train_test_split_point)
test.set_window(start=train_test_split_point, end=end_date)

# preprocessing 
first = create_df(train.buildings[1].elec.mains())
# preprocessing 1
main_train_df = create_df(train.buildings[1].elec.mains())
main_train_df = main_train_df.where(main_train_df >= 0).fillna(method='ffill')
main_test_df = create_df(test.buildings[1].elec.mains())
main_test_df = main_test_df.where(main_test_df >= 0).fillna(method='ffill')
# preprocessing 2
# neg_mask = first.where(first < 0)
# print(neg_mask)
# neg_mask = neg_mask.fillna(0)
# df2_diff = neg_mask.diff()
# main_df2 = main_df.copy() + df2_diff
# draw_plot([neg_mask, df2_diff, main_df2], title="Negative values and their difference")
# corr_list = [main_df2, first, main_df]
# print(corr_list)
corr_list = [main_train_df, first]
# draw_plot(corr_list, title="Main corrected")
# Training plots
# train_test_mains = [create_df(train.buildings[1].elec.mains()), create_df(test.buildings[1].elec.mains())]
# draw_plot(train_test_mains, title="Trainset & Testset Mains")

total_sum = 0
dataset_df_list = []
dataset_df_list_corrected = []
for meter in dataset.buildings[1].elec.submeters().meters:
    df = create_df(meter)
    dataset_df_list.append(df)
    total_sum += df.sum().sum()
    df_corrected = df.where(df >= 0).fillna(method='ffill')
    dataset_df_list_corrected.append(df_corrected)
dataset_df = create_df(dataset.buildings[1].elec.mains())
dataset_df_corrected = dataset_df.where(dataset_df >= 0).fillna(method='ffill')

sum_of_submeters = sum(dataset_df_list)
difference_df = sum_of_submeters - create_df(dataset.buildings[1].elec.mains())

print()
print("Percentage of Energy Submetered: ", total_sum / create_df(dataset.buildings[1].elec.mains()).sum().sum() * 100)
draw_plot(dataset_df_list_corrected, title="Dataset Submeters", metergroup=dataset.buildings[1].elec)



comparison_list = [sum_of_submeters, dataset_df_corrected]
draw_plot(comparison_list, title="Sum of Submeters & Site Meter")

# print("*******************************")
# print(difference_df.sum())
# draw_plot([difference_df], title="Difference")

# Create submeter list
train_list = []
test_list =[]
for train_meter, test_meter in zip(train.buildings[1].elec.submeters().meters, test.buildings[1].elec.submeters().meters):
    train_df = create_df(train_meter)
    train_df_corrected = train_df.where(train_df >= 0).fillna(method='ffill')
    train_list.append(train_df_corrected)
    test_df = create_df(test_meter)
    test_df_corrected = test_df.where(test_df >= 0).fillna(method='ffill')
    test_list.append(test_df_corrected)

# Aggregate submeters
train_sum = sum(train_list)
train_sum_sum = train_sum.sum()
test_sum = sum(test_list)
test_sum_sum = test_sum.sum()

# Plot submeters + aggregate
# draw_plot(train_list, title="Submeters Trainset", metergroup=train.buildings[1].elec)
# draw_plot(sum_train, title="Sum of Submeters Trainset")
# draw_plot(test_list, title="Submeters Testset", metergroup=train.buildings[1].elec)
# draw_plot(sum_test, title="Sum of Submeters Testset")

train_list = [train_sum, main_train_df]
test_list = [test_sum, main_test_df]

# Plot aggregate vs Site Meter
draw_plot(train_list, title="Sum of Submeters & Site Meter Trainset")
draw_plot(test_list, title="Sum of Submeters & Site Meter Testset")




# stacked plots sind scheiße
# draw_stackedplot(train_list, title="Aggregated Meters")
# a_list = [main_df]
# a_list.extend(train_list)
# draw_stackedplot(a_list, title="Aggregated Meters & SiteMeter")

# train_list_comp = train_list.copy()
# train_list_comp.append(create_df(train.buildings[1].elec.mains()))
# draw_stackedplot(train_list_comp, title="aggregate vs site meter")

# draw_plot(aggregated_df, title="Unlabeled Dataset")


# Create main train and test for disaggregation
"""

Wird oben schon erstellt 

"""
train_df = create_df(train.buildings[1].elec.mains()) # power_series_all_data() -> series.Series  ,   to_frame() -> frame.DataFrame
train_main = [train_df]
test_df = create_df(test.buildings[1].elec.mains())
test_corrected = test_df.where(test_df >= 0).fillna(method='ffill')
test_main = [test_corrected]


# create list of all meters in test dataset
test_dataframe_list = []
for meter in test.buildings[1].elec.submeters().meters:
    df = create_df(meter)
    df_corrected = df.where(df >= 0).fillna(method='ffill')
    appliance_type = meter.label()
    df_corrected.columns = [appliance_type]
    test_dataframe_list.append(df_corrected)


# find out top k meters
train_not_top_k = train.buildings[1].elec.submeters().select_top_k(k=12)
train_top_k = train.buildings[1].elec.submeters().select_top_k(k=8)

remaining_meters = [meter for meter in train.buildings[1].elec.submeters().meters if meter not in train_top_k.meters]

print("----------------------------------------------------------------------------------------------------")
print("train_top_k: ", train_top_k)
print("----------------------------------------------------------------------------------------------------")
print("remaining_meters: ", remaining_meters)

train_top_k_list = []
remaining_meters_list = []
train_top_k_sum_sum = 0
remaining_meters_sum_sum = 0

for meter in train_top_k.meters:
    df = create_df(meter)
    train_top_k_list.append(df)
    train_top_k_sum_sum += df.sum().sum()

for meter in remaining_meters:
    df = create_df(meter)
    remaining_meters_list.append(df)
    remaining_meters_sum_sum += df.sum().sum()


print("----------------------------------------------------------------------------------------------------")
print("train_top_k_list: ", train_top_k_list)
print("----------------------------------------------------------------------------------------------------")
print("train_not_top_k_list: ", remaining_meters_list)
print("----------------------------------------------------------------------------------------------------")
print("train_top_k_sum_sum: ", train_top_k_sum_sum)
print("train_not_top_k_sum_sum: ", remaining_meters_sum_sum)

# whole_data = create_df(train.buildings[1].elec.mains()).sum().sum()
coverage_percentage = (train_top_k_sum_sum / train_sum_sum * 100).iloc[0]
print("------------------------------------------")
print(f"coverage_percentage: {coverage_percentage:.2f}%")
print("------------------------------------------")


# get the indices
top_k_instances = [meter.instance() for meter in train_top_k.meters]
print("top_k_instances: ", top_k_instances)
train_appliances = []
for i in top_k_instances:
    appliance = train.buildings[1].elec[i]
    appliance_name = appliance.label()
    appliance_df = create_df(appliance)
    corrected_df = appliance_df.where(appliance_df >= 0).fillna(method='ffill')
    appliance_data = [appliance_df]
    # draw_plot(appliance_data, title=f"{appliance_name} corrected")

    existing_names = [name for name, _ in train_appliances]
    if appliance_name in existing_names:
        count = sum(1 for name in existing_names if name.startswith(appliance_name))
        appliance_name = f"{appliance_name}_{count + 1}"
        
    train_appliances.append((appliance_name, appliance_data))


gt_list = []
for i in top_k_instances:
    index = i - 1   # instance=1 is at entry 0
    gt_list.append(test_dataframe_list[index])

draw_plot(gt_list, "Ground Truth")


# FHMM disaggregation
fhmm = FHMMExact({})    # 1 n Elemente als Input -> n Elemente als Output
fhmm.partial_fit(train_main=train_main, train_appliances=train_appliances)
fhmm_prediction_list = fhmm.disaggregate_chunk(test_main)   # list of dataframes (nur ein Eintrag)
draw_plot(fhmm_prediction_list, title="FHMM Disaggregation", metergroup=train.buildings[1].elec, lim="FHMM", top_k=top_k_instances)

# CO disaggregation
co = CO({})
co.partial_fit(train_main=train_main, train_appliances=train_appliances)
co_prediction_list = co.disaggregate_chunk(test_main)
draw_plot(co_prediction_list, title="CO Disaggregation", metergroup=train.buildings[1].elec, lim="CO", top_k=top_k_instances)

# Mean disaggregation
mean = Mean({})
mean.partial_fit(train_main=train_main, train_appliances=train_appliances)
mean_prediction_list = mean.disaggregate_chunk(test_main)
# draw_plot(mean_prediction_list)





test_tuple_list = []

gt_dict = {index: test_dataframe_list[index - 1] for index in top_k_instances}

fhmm_df = fhmm_prediction_list[0]
fhmm_predictions = [fhmm_df[[col]] for col in fhmm_df.columns]
co_df = co_prediction_list[0]
co_predictions = [co_df[[col]] for col in co_df.columns]
# mean_df = mean_prediction_list[0]
# mean_predictions = [mean_df[[col]] for col in mean_df.columns]
''' predictions in order of highest to lowest power
    gt_dict in order of top_10_instances that rank from highest to lowerst power
'''
print("fhmm f1_score:")
print(mymetrics.f1_score(fhmm_predictions, gt_dict))
print("co f1_score:")
print(mymetrics.f1_score(co_predictions, gt_dict))
print("**************************************************************************************")
print("fhmm nmae_normalized_error_power:")
print(mymetrics.normalized_mean_absolute_error_power(fhmm_predictions, gt_dict))
print("co nmae_normalized_error_power:")
print(mymetrics.normalized_mean_absolute_error_power(co_predictions, gt_dict))
print("**************************************************************************************")
print("fhmm mean_absolute_error_power:")
print(mymetrics.mean_absolute_error_power(fhmm_predictions, gt_dict))
print("co mean_absolute_error_power:")
print(mymetrics.mean_absolute_error_power(co_predictions, gt_dict))
print("**************************************************************************************")
print("fhmm rms_error_power:")
print(mymetrics.rms_error_power(fhmm_predictions, gt_dict))
print("co rms_error_power:")
print(mymetrics.rms_error_power(co_predictions, gt_dict))
print("**************************************************************************************")


fhmm_list = []
co_list = []
gt_list = []
for fhmm, co, mean, gt in zip(fhmm_prediction_list[0], co_prediction_list[0], mean_prediction_list[0], top_k_instances):
    # in order of highest to lowest power (natural order of fhmm_prediction_list and top_10_instances)
    index = gt - 1  # the first index of gt is 0 but i want to compare to the instance number and not the index - the indices go from 2 to 21
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    co_df = co_prediction_list[0][co].to_frame()
    # mean_df = mean_prediction_list[0][mean].to_frame()
    fhmm_list.append(fhmm_df)
    co_list.append(co_df)
    gt_list.append(test_dataframe_list[index])
    # fhmm_df.index = pd.date_range(start='2013-01-02 00:00:00', end='2013-01-02 23:59:00', freq='1T', tz='Europe/Berlin')
    # df_list = [fhmm_df, co_df, mean_df, test_dataframe_list[index]]
    df_list = [fhmm_df, co_df, test_dataframe_list[index]]

    fhmm_df.rename(columns=lambda x: x.split('_')[0], inplace=True)
    co_df.rename(columns=lambda x: x.split('_')[0], inplace=True)

    title = fhmm_df.columns[0]
    draw_plot2(df_list, title=title)


fhmm_prediction_hom_indices = []
for i in fhmm_prediction_list[0]:
    df = fhmm_prediction_list[0][i].to_frame()
    df.columns = pd.MultiIndex.from_tuples([("power", "reactive")])
    fhmm_prediction_hom_indices.append(df)
fhmm_prediction_sum = sum(fhmm_prediction_hom_indices)



co_prediction_hom_indices = []
for i in co_prediction_list[0]:
    df = co_prediction_list[0][i].to_frame()
    df.columns = pd.MultiIndex.from_tuples([("power", "reactive")])
    co_prediction_hom_indices.append(df)
co_prediction_sum = sum(co_prediction_hom_indices)



# draw_plot([fhmm_prediction_sum], title="FHMM aggregated")
# draw_plot([co_prediction_sum], title="CO aggregated")
fhmm_prediction_sum.columns = ["FHMM"]
co_prediction_sum.columns = ["CO"]
test_corrected.columns = ["GT"]
test_sum.columns = ["GT"]


draw_plot([fhmm_prediction_sum, co_prediction_sum, test_corrected], title="Predictions & Site Meter")
draw_plot([fhmm_prediction_sum, co_prediction_sum, test_sum], title="Predictions & aggregated GT")

draw_plot([test_sum, test_corrected], title="Sum of Submeters & Site Meter")

# draw_stackedplot(fhmm_list, title="Stacked FHMM")
# draw_stackedplot(co_list, title="Stacked CO")
# draw_stackedplot(gt_list, title="Stacked GT")