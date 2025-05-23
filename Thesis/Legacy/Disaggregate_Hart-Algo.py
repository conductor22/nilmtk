from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib as plt
import pandas as pd
import numpy as np
from six import iteritems
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
import nilmtk.utils
from pprint import pprint

rcParams['figure.figsize'] = (13, 6)

# AMPds data set ist aus dem Internet
data = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")
print("Loaded", len(data.buildings), "buildings")

elec = data.buildings[1].elec
print(elec.get_timeframe())

data.set_window(start="2014-01-01", end="2014-01-03")

mains = elec.mains()
subs = elec.submeters()

from nilmtk.legacy.disaggregate.hart_85 import Hart85
h = Hart85()
#pprint(dir(mains))
pprint(f"mains load: {mains.load()}")
#pprint(subs)
print(f"mains timeframe: {mains.get_timeframe()}")
#print(subs.get_timeframe())

data.buildings[1].elec.plot()
plt.pyplot.show()

h.train(mains, columns=[("power", "active")])
output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/refith5/output.h5", "w")
df = h.disaggregate(mains, output)

print(df[0].index.min(), df[0].index.max())
print(df[1].index.min(), df[1].index.max())
'''
from nilmtk.legacy.disaggregate import CombinatorialOptimisation
from nilmtk import MeterGroup
mains_list = [mains]
mains_group = MeterGroup(mains_list)

co = CombinatorialOptimisation()
co.train(mains_group, columns=[("power", "active")])
df_co = co.disaggregate(mains_group, output)

df_co.plot()
plt.pyplot.show()
'''

df.plot()
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()

pprint(df.tail())
pprint(df.head())
pprint(elec)
h.best_matched_appliance(subs, df)
print(df.index.min())



df_freezer = next(elec[4].load())
merged_df = pd.merge(df[0], df_freezer, left_index=True, right_index=True)
ax1 = merged_df[0].plot(c='r')
ax2 = merged_df['power', 'active'].plot(c='grey')
ax1.legend(["Predicted", "Ground truth"])
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()

df_freezer = next(elec[7].load())
merged_df = pd.merge(df[1], df_freezer, left_index=True, right_index=True)
ax1 = merged_df[1].plot(c='r')
ax2 = merged_df['power', 'active'].plot(c='grey')
ax1.legend(["Predicted", "Ground truth"])
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()


df_freezer = next(elec[10].load())
merged_df = pd.merge(df[2], df_freezer, left_index=True, right_index=True)

ax1 = merged_df[2].plot(c='r')
ax2 = merged_df['power', 'active'].plot(c='grey')
ax1.legend(["Predicted", "Ground truth"])
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()

df_freezer = next(elec[8].load())
merged_df = pd.merge(df[3], df_freezer, left_index=True, right_index=True)

ax1 = merged_df[3].plot(c='r')
ax2 = merged_df['power', 'active'].plot(c='grey')
ax1.legend(["Predicted", "Ground truth"])
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()
'''
df_dish_washer = next(elec[2].load())
merged_df = pd.merge(df[1], df_dish_washer, left_index=True, right_index=True)

pprint(merged_df.head())

merged_df[1].plot(c='r')
merged_df['power', 'active'].plot()
plt.pyplot.legend(["Predicted", "Ground truth"])
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()

df_freezer = next(elec[10].load())
merged_df = pd.merge(df[0], df_freezer, left_index=True, right_index=True)

ax1 = merged_df[0].plot(c='r')
ax2 = merged_df['power', 'active'].plot(c='grey')
ax1.legend(["Predicted", "Ground truth"])
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()
'''
print(df_dish_washer.index.min(), df_dish_washer.index.max())
print(df_freezer.index.min(), df_freezer.index.max())

print(df[0].index.min(), df[0].index.max())
print(df[1].index.min(), df[1].index.max())

'''
data.buildings[1].elec.plot()
both = pd.merge(df, data.buildings[1].elec).plot()
plt.pyplot.ylabel("Power (W)")
plt.pyplot.xlabel("Time")
plt.pyplot.show()


fig, ax = plt.pyplot.subplots(figsize=(13, 6))
data.buildings[1].elec.plot(ax=ax)
df.plot(ax=ax)
ax.legend()
plt.pyplot.show()

import matplotlib.gridspec as gridspec

fig = plt.pyplot.figure(figsize=(13, 10))

gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # The first plot takes more space

ax1 = plt.pyplot.subplot(gs[0])
data.buildings[1].elec.plot(ax=ax1)
ax1.set_title('Devices in Building 1')

ax2 = plt.pyplot.subplot(gs[1])
df.plot(ax=ax2)
ax2.set_title('DataFrame Plot')

plt.pyplot.tight_layout()  # Adjust the spacing between subplots
plt.pyplot.show()
'''