from nilmtk import DataSet, MeterGroup # kein nilmtk. vor DataSet/MeterGroup nötig
from nilmtk.dataset_converters import convert_refit
from pprint import pprint as pp
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import nilmtk

plt.style.use("ggplot")
rcParams["figure.figsize"] = (13, 10)

refit_ds = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")

# pp(refit_ds.buildings[1].elec.mains().power_series_all_data().head())

pp(refit_ds.buildings[1].elec)   # meter group of building 1
# pp(refit_ds.buildings[1].elec.mains().total_energy())
# pp(refit_ds.buildings[1].elec.submeters().energy_per_meter())
# pp(refit_ds.buildings[1].elec.submeters().total_energy())

# fraction = refit_ds.buildings[1].elec.submeters().fraction_per_meter().dropna()
# labels = refit_ds.buildings[1].elec.get_labels(fraction.index)
plt.figure(figsize=(10,30))
# fraction.plot(kind="pie", labels=labels)
# plt.show()
# refit_ds.buildings[1].elec.draw_wiring_graph()
# plt.show()
# refit_ds.buildings[1].elec.plot_when_on(on_power_threshold = 40)
# plt.show()

#refit_ds.set_window(start="2014-01-01", end="2014-12-31")
refit_ds.buildings[1].elec[3].plot()
plt.show()