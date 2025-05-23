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

# import nilmtk_contrib
# print(nilmtk_contrib.__version__)

# Datensets erstellen
dataset = DataSet("C:/Users/ieh-buergin/Desktop/eshl.h5")
train = DataSet("C:/Users/ieh-buergin/Desktop/eshl.h5")
test = DataSet("C:/Users/ieh-buergin/Desktop/eshl.h5")

df = dataset.buildings[1].elec.mains().power_series_all_data().to_frame()

# start_date = df.index[0].date()
# end_date = df.index[-1].date()+
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

test_elec = test.buildings[1].elec.submeters()
all_meters = [test.buildings[1].elec.mains(), test.buildings[1].elec.submeters()]
print(all_meters)
draw_plot(all_meters, "main & submeters")
# aggregate = train.buildings[1].elec.submeters().power_series_all_data().to_frame()

test_list = []
for meter in test_elec.meters:
    df = meter.power_series_all_data().to_frame()
    test_list.append(df)
test_list.append(test.buildings[1].elec.mains().power_series_all_data().to_frame())
draw_plot(test_list, "test main & submeters")


aggregate = test_elec.meters[0].power_series_all_data().to_frame()
for meter in test_elec.meters:
    if meter.instance() == "1":
        print("skipped")
        continue
    print(meter.instance())
    df = meter.power_series_all_data().to_frame()
    df.columns = pd.MultiIndex.from_tuples([("power", "active")])
    aggregate += df
whatever_list = [aggregate, test.buildings[1].elec.mains()]
draw_plot(whatever_list, "aggregate")

# Main train and test
train_df = train.buildings[1].elec.mains().power_series_all_data().to_frame()
train_main = [train_df]
test_df = test.buildings[1].elec.mains().power_series_all_data().to_frame()
test_main = [test_df]


# find out top k meters
top_10 = train.buildings[1].elec.submeters().select_top_k(k=7) # Metergroup

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


import gc
import time

params = {'num_of_states': 3}
fhmm = FHMMExact(params)

def chunk_data(data, chunk_size):
    """Split data into chunks of size chunk_size."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

chunk_size = 100000

# Split the train_main and train_appliances into chunks
train_main_chunks = list(chunk_data(train_main, chunk_size))
train_appliances_chunks = list(chunk_data(train_appliances, chunk_size))

# Iterate over each chunk and call partial_fit
for train_main_chunk, train_appliances_chunk in zip(train_main_chunks, train_appliances_chunks):
    fhmm.partial_fit(train_main=train_main_chunk, train_appliances=train_appliances_chunk)

fhmm.partial_fit(train_main=train_main, train_appliances=train_appliances)



print("****************************")
print("trying to disaggregate")
print("****************************")
start = time.time()

chunk_size = 100000
fhmm_prediction_list = []

for start in range(0, len(test_main[0]), chunk_size):
    print(len(test_main[0]))
    end = start + chunk_size
    test_main_chunk = test_main[0][start:end]
    print(start, end)
    chunk_prediction = fhmm.disaggregate_chunk([test_main_chunk])
    fhmm_prediction_list.append(chunk_prediction[0])

    gc.collect()
fhmm_prediction_list = pd.concat(fhmm_prediction_list)
fhmm_prediction_list = [fhmm_prediction_list]
print("****************************")
print("disaggregation done")
print("****************************")
end = time.time()
total = end - start
minutes = int(total/60)
seconds = total%60
print(f"Total time: {minutes}m{seconds:.0f}s")

draw_plot(fhmm_prediction_list)

params2 = {'num_of_states': 2}
fhmm2 = FHMMExact(params2)
fhmm2.partial_fit(train_main=train_main, train_appliances=train_appliances)
fhmm2_prediction_list = fhmm2.disaggregate_chunk(test_main)   # list of dataframes (nur ein Eintrag)
draw_plot(fhmm2_prediction_list)

params3 = {'num_of_states': 1}
fhmm3 = FHMMExact(params3)
fhmm3.partial_fit(train_main=train_main, train_appliances=train_appliances)
fhmm3_prediction_list = fhmm3.disaggregate_chunk(test_main)   # list of dataframes (nur ein Eintrag)
draw_plot(fhmm3_prediction_list)

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

for fhmm, co, mean, gt in zip(fhmm_prediction_list[0], fhmm2_prediction_list[0], fhmm3_prediction_list[0], top_10_instances):
    index = gt - 2  # the first index of gt is 0 but i want to compare to the instance number and not the index - the indices go from 2 to 21
    fhmm_df = fhmm_prediction_list[0][fhmm].to_frame()
    fhmm2_df = fhmm2_prediction_list[0][co].to_frame()
    fhmm3_df = fhmm3_prediction_list[0][mean].to_frame()
    # fhmm_df.index = pd.date_range(start='2013-01-02 00:00:00', end='2013-01-02 23:59:00', freq='1T', tz='Europe/Berlin')
    df_list = [fhmm_df, fhmm2_df, fhmm3_df, test_dataframe_list[index]]
    # draw_plot(fhmm_df)
    # draw_plot(test_dataframe_list[index])
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
