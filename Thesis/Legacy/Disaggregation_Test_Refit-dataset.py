from nilmtk import DataSet, HDFDataStore
from nilmtk.metrics import f1_score
#from nilmtk.dataset_converters import
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pprint import pprint

# refit data set ist aus dem internet
dataset = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")

train = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")
test = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")
dataset.set_window(start="2014-04-01", end="2014-10-01")
train.set_window(start="2014-04-01", end="2014-07-01")
test.set_window(start="2014-07-01", end="2014-10-01")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

pprint(dir(train.buildings[1].elec))
pprint(dir(train.buildings[1].elec.identifier))
# print(dataset.buildings[1].elec.get_timeframe())    # TimeFrame(start='2013-10-09 14:06:17+01:00', end='2015-07-10 12:56:32+01:00', empty=False)
# print(train.buildings[1].elec.get_timeframe())      # TimeFrame(start='2014-03-10 00:00:00+00:00', end='2014-03-25 00:00:00+00:00', empty=False)
# print(test.buildings[1].elec.get_timeframe())       # TimeFrame(start='2014-03-25 00:00:00+00:00', end='2014-04-01 00:00:00+01:00', empty=False)
plt.figure(figsize=(18, 10))

plt.subplot(3, 1, 1)
dataset.buildings[1].elec.plot()
plt.title("Dataset")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend(loc="upper left")

plt.subplot(3, 1, 2)
train_elec.plot()
plt.title("Training Data")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend().remove()

plt.subplot(3, 1, 3)
test_elec.plot()
plt.title("Test Data")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend().remove()

plt.show()




#dataset.buildings[1].elec.plot()
#plt.show()



fhmm = FHMM()
fhmm.train(train.buildings[1].elec)
fhmm_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/refith5/fhmm.h5", "w")

fhmm.disaggregate(test.buildings[1].elec.mains(), fhmm_output)
fhmm_output.close()
print("fhmm done")
'''
co = CombinatorialOptimisation()
co.train(train.buildings[1].elec)
co_output = HDFDataStore("C:/Users/Megapoort/Desktop/nilmdata/refith5/co.h5", "w")

co.disaggregate(test.buildings[1].elec.mains(), co_output)
co_output.close()
print("co done")
'''
ground_truth = test_elec

fhmm_predictions = DataSet('C:/Users/Megapoort/Desktop/nilmdata/refith5/fhmm.h5')
print("f1_score fhmm")
print(f1_score(ground_truth, fhmm_predictions))

co_predictions = DataSet('C:/Users/Megapoort/Desktop/nilmdata/refith5/co.h5')
print("f1_score co")
print(f1_score(ground_truth, co_predictions))