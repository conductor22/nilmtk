from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
import matplotlib.pyplot as plt
from pprint import pprint
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk.metrics import f1_score
import nilmtk.utils
from matplotlib import rcParams



dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")
dataset_elec = dataset.buildings[1].elec

train = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")
test = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/AMPds2.h5")

dataset.set_window(start="2013-01-01", end="2013-03-01")
train.set_window(start="2013-01-01", end="2013-02-01")
test.set_window(start="2013-02-01", end="2013-03-01")

'''
fig, axs = plt.subplots(2, 2, figsize=(20,10))

dataset.buildings[1].elec.plot(ax=axs[0 ,0])
axs[0, 0].set_title("whole data")
axs[0, 0].legend().remove()



dataset.buildings[1].elec.plot(ax=axs[1 ,0])
axs[0, 1].set_title("windowed data")
axs[0, 1].legend().remove()

train.buildings[1].elec.plot(ax=axs[0 ,1])
axs[1, 0].set_title("train data")
axs[1, 0].legend().remove()
test.buildings[1].elec.plot(ax=axs[1 ,1])
axs[1, 1].set_title("test data")
axs[1, 1].legend().remove()

plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.show()
'''
top_5_train_elec = train.buildings[1].elec.submeters().select_top_k(k=2)
print(top_5_train_elec)

top_5_train_elec.plot()
plt.legend("top 5 elecs")
plt.show()
test.buildings[1].elec.mains().plot()
plt.legend("site meter")
plt.show()

fhmm = FHMM()
fhmm.train(top_5_train_elec)
fhmm_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5", "w")

print("*******************************************")
print("fhmm training done")
print("*******************************************")

# fhmm_output.close()
# fhmm_test = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5")
# fhmm_test.plot()
# plt.show()

fhmm.disaggregate(test.buildings[1].elec.mains(), fhmm_output)
fhmm_output.close()

print("*******************************************")
print("fhmm disaggregation done")
print("*******************************************")


co = CombinatorialOptimisation()
co.train(top_5_train_elec)
co_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5", "w")

print("*******************************************")
print("co training done")
print("*******************************************")

co.disaggregate(test.buildings[1].elec.mains(), co_output)
co_output.close()

print("*******************************************")
print("co disaggregation done")
print("*******************************************")





ground_truth = test.buildings[1].elec

fhmm_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/fhmm.h5")
co_dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/ampds/co.h5")
fhmm_predictions = fhmm_dataset.buildings[1].elec
co_predictions = co_dataset.buildings[1].elec

print("FHMM f1-score:")
print(f1_score(ground_truth=ground_truth, predictions=fhmm_predictions))
print("CO f1-score:")
print(f1_score(ground_truth=ground_truth, predictions=co_predictions))

# print("FHMM time window")
# print(fhmm_predictions.index.min(), "-", fhmm_predictions.index.max())
# print("CO time window")
# print(co_predictions.index.min(), "-", co_predictions.index.max())

fhmm_predictions.plot()
plt.legend("fhmm predictions")
plt.show()
co_predictions.plot()
plt.legend("co predictions")
plt.show()
ground_truth.plot()
plt.legend("ground truth")
plt.show()


indices = [18, 14]


for i, index in enumerate(indices):
    device = ground_truth[index]
    fhmm_device_predictions = fhmm_predictions[index]
    co_device_predictions = co_predictions[index]

    rcParams["figure.figsize"] = (16, 8)
    device.plot()
    fhmm_device_predictions.plot()
    co_device_predictions.plot()
    plt.title(f"Device {index}")
    plt.legend(["Ground Truth", "FHMM", "CO"])
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.grid()
    plt.tight_layout()
    plt.show()
'''
device = ground_truth[20]
fhmm_device_predictions = fhmm_predictions[20]
co_device_predictions = co_predictions[20]

plt.figure(figsize=(15, 8))
device.plot()
fhmm_device_predictions.plot()
co_device_predictions.plot()

plt.legend(["Ground Truth", "FHMM", "CO"])
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.grid()
plt.tight_layout()
plt.show()
'''

