from nilmtk import DataSet
from nilmtk.dataset_converters import convert_refit

# convert_refit("C:/Users/Megapoort/Desktop/nilmdata/refitcsv", "C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")
# print("converted")
refit_ds = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")

from pprint import pprint
'''
pprint(vars(refit_ds))
print("*************************")
# print(refit_ds)
# print("*************************")
# for key, value in refit_ds.buildings.items():
#     print(f"{key} : {value}")
# print("*************************")
# print(refit_ds.buildings[1].elec)
pprint(refit_ds.buildings)
print("*************************")
pprint(refit_ds.metadata)
print("*************************")
print(vars(refit_ds.buildings[1]))
print("*************************")
pprint(refit_ds.buildings[1].elec)
print("*************************")
pprint(refit_ds.buildings[1].metadata)
'''
pprint(refit_ds.buildings[1].elec)
elec = refit_ds.buildings[1].elec
pprint(elec[1])
pprint(elec[1].available_columns())
df = next(elec[1].load(ac_type="active", sample_period=60))
print(df.head())

from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import nilmtk
from nilmtk import DataSet, MeterGroup

# import numpy as np
# x=np.linspace(0,10,100)
# y=np.sin(x)
# plt.plot(x,y)
# plt.show()

df.index = pd.to_datetime(df.index)
plt.figure(figsize=(10,6))
plt.plot(df.index, df["power"], label="Active Power", color="b")
plt.xlabel=("time")
plt.ylabel("power (W)")
plt.title("power over time")
plt.legend()
plt.grid(True)
plt.show()

# plt.style.use("ggplot")
# rcParams["figure.figsize"] = (13,10)
# print()
# # print("nested metergroups")
# # pprint(elec.nested_metergorups())
# print()
# print("mains")
# mains = elec.mains()
# pprint(mains)
# print()
# print("submeters")
# pprint(elec.submeters())
# print()
# print("proportion of energy submetered")
# elec.proportion_of_energy_submetered()
# print()

# print()
# print("mains available ac types")
# pprint(mains.available_ac_types("power"))
# print()
# print("submeters available ac types")
# pprint(elec.submeters().available_ac_types("power"))
# print()
# print("load")
# pprint(next(elec.load()))
# print()
# print("total energy in kWh")
# pprint(elec.mains().total_enery())
# print()
# print("energy per meter in kWh")
# pprint(elec.submeters().energy_per_meter())
'''
refit_ds.set_window(start="2021-01-01", end="2021-01-02")
print()
print("windowed")
print()
train_building = refit_ds.buildings[1]
test_building = refit_ds.buildings[2]
print("sets assigned")
print()

from nilmtk.disaggregate import Hart85
disaggregator = Hart85()
print("aggregator instantiated")
print()
disaggregator.train(train_building)
print("train()")
print()
disaggregated_data = disaggregator.disaggregate(test_building)
print("disaggregated")
# ground_truth_data = test_building.elec
# print(rmse(disaggregated_data, ground_truth_data))
'''