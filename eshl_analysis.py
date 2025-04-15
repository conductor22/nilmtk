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

def create_df(meter):
    df_active_power = meter.power_series_all_data(ac_type='active').to_frame()
    # df_reactive_power = meter.power_series_all_data(ac_type='reactive').to_frame()
    # df = pd.concat([df_active_power, df_reactive_power], axis=1)
    df = df_active_power
    return df
 

# Datensets erstellen
dataset = DataSet("C:/Users/ieh-buergin/Desktop/allcsv/eshl.h5") # ist nur p1
train = DataSet("C:/Users/ieh-buergin/Desktop/allcsv/eshl.h5")
test = DataSet("C:/Users/ieh-buergin/Desktop/allcsv/eshl.h5")

start_date = pd.Timestamp("2024-08-02")
end_date = pd.Timestamp("2024-08-31")
dataset.set_window(start=start_date, end=end_date)



dataset_main_df = create_df(dataset.buildings[1].elec.mains())
dataset_sub_df_list = []
for meter in dataset.buildings[1].elec.submeters().meters:
    df = create_df(meter)
    dataset_sub_df_list.append(df)

aggregated_submeters = dataset_sub_df_list[0].copy()
for i in dataset_sub_df_list[1:]:
    aggregated_submeters += i

dropoutrate = 1 - len(dataset_main_df)/len(pd.date_range(start=start_date, end=end_date, freq='s'))
print("df length: ", len(dataset_main_df))
print("date range length: ", len(pd.date_range(start=start_date, end=end_date, freq='s')))
print("droputrate: ", dropoutrate)

print("---------------------")
for df in dataset_sub_df_list:
    dropoutrate = 1 - len(df)/len(pd.date_range(start=start_date, end=end_date, freq='s'))
    print("dropout rate: ", dropoutrate)

expected_time_range = pd.date_range(start=start_date, end=end_date, freq='s')
expected_time_range = expected_time_range.tz_localize('CET')
print(dataset_main_df.index)
print("************************")
print(expected_time_range)


missing_timestamps = expected_time_range.difference(dataset_main_df.index)
print(missing_timestamps)

plt.figure(figsize=(10, 6))
plt.plot(expected_time_range, [1 if ts in missing_timestamps else 0 for ts in expected_time_range],
         label="Missing Timestamps")

# Add labels, title, and grid
plt.title('Missing Timestamps Visualization')
plt.xlabel('Time')
plt.ylabel('Missing (1) / Present (0)')
plt.legend()
plt.grid(True)
plt.show()
fehler

dataset_main_df = dataset_main_df.where(dataset_main_df >= 0).fillna(method='ffill')

comparison_list = [dataset_main_df, aggregated_submeters]
draw_plot(comparison_list, "Site Meter and Aggregated Submeters P1")


difference_df = dataset_main_df - aggregated_submeters

total_site_meter_energy = dataset_main_df.sum()
total_submeter_energy = aggregated_submeters.sum()
submeter_percentage = (total_submeter_energy / total_site_meter_energy) * 100
print("submeter percentage: ", submeter_percentage)

draw_plot(difference_df, "Difference Site Meter and Aggregated Submeters P1")

print(dataset_main_df.head())
print("************************")
print(dataset_main_df.tail())

draw_plot(dataset_main_df, "Dataset Site Meter P1")
draw_plot(dataset_sub_df_list, "Dataset Submeters P1", train_elec=dataset.buildings[1].elec.submeters())