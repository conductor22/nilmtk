from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from matplotlib import rcParams
from plotting import draw_plot
import pandas as pd
'''
zur bewertung der disaggregation muss erstmal geschaut werden,
ob die aggregierten verbräuche vor der disaggregation dem gesamtverbrauch entsprechen
'''
dataset = DataSet("C:/Users/ieh-buergin/Desktop/eshl/eshl.h5")
# dataset.set_window(start="2024-08-01", end="2024-08-03")  # komischer Strich
# dataset.set_window(start="2024-08-01", end="2024-08-02")  # kein Strich
# dataset.set_window(start="2024-08-02", end="2024-08-03")  # kein Strich
dataset.set_window(start="2024-08-02", end="2024-09-01")
elec = dataset.buildings[1].elec
mains = elec.mains()
subs = elec.submeters()

# WTFFFFFFFF
mains_df = mains.power_series_all_data().to_frame()
print(mains_df.head())
data_generator = mains.load()
first_chunk = next(data_generator)
print(type(first_chunk))    # dataframe
print(first_chunk.head())
df_list = [mains_df, mains]
draw_plot(df_list, title="df und mains")
draw_plot(subs, title="subs")

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

df_all = [meter_1_to_12, mains]

draw_plot(df_all, title="meter_1_to_12, haushalt, wiz01")