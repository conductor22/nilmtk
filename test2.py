from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd

dataset = DataSet("E:/Users/Megapoort/eshldaten/csv/eshl.h5")
dataset.set_window(start="2024-08-01", end="2024-08-02")
elec = dataset.buildings[1].elec

print(elec)

df_list = []
for meter in elec.meters:
    df = meter.power_series_all_data().to_frame()
    df_list.append(df)

print(len(df_list))
# meterclamps zusammenrechnen
meter_1_to_12 = df_list[0].copy()
meter_1_to_12[:] = 0
for i in range(0, 12):
    meter_1_to_12 += df_list[i]

# "HH" ausrechnen - wiz01 - wiz02...07
wiz01 = df_list[12]
haushalt = wiz01.copy()
for i in range(13, 19):
    haushalt -= df_list[i]

df_all = [meter_1_to_12, haushalt, wiz01]

draw_plot(df_all, title="meter_1_to_12, haushalt, wiz01")