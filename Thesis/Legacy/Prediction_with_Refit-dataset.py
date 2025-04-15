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

rcParams["figure.figsize"] = (13, 6)
train = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")
test = DataSet("C:/Users/Megapoort/Desktop/nilmdata/refith5/refit.h5")


train.set_window(start="2014-03-09", end="2014-03-20")
test.set_window(start="2014-03-20", end="2014-04-01")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

# train_elec.plot()
# plt.pyplot.show()
# test_elec.mains().plot()
# plt.pyplot.show()

# top_5_train_elec = train_elec.submeters().select_top_k(k=5)
tran_elec_all = train_elec.submeters()

def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt = {}

    for i, chunk in enumerate(test_elec.mains().load(physical_quantity = "power", ac_type = "active", sample_period = sample_period)):
        chunk_drop_na = chunk.dropna()
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        gt[i]={}

        for meter in test_elec.submeters().meters:
            gt[i][meter] = next(meter.load(physical_quantity = 'power', ac_type = 'active', sample_period=sample_period))
        gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i]) if len(v)}, index=next(iter(gt[i].values())).index).dropna()

    gt_overall = pd.concat(gt)
    gt_overall.index = gt_overall.index.droplevel()
    pred_overall = pd.concat(pred)
    pred_overall.index = pred_overall.index.droplevel()

    gt_overall = gt_overall[pred_overall.columns]

    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)

    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.loc[common_index_local]
    pred_overall = pred_overall.loc[common_index_local]
    appliance_labels = [m for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall

classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}
predictions = {}
sample_period = 30
for clf_name, clf in classifiers.items():
    print("*"*20)
    print(clf_name)
    print("*" *20)
    start = time.time()
    # Note that we have given the sample period to downsample the data to 1 minute. 
    # If instead of top_5 we wanted to train on all appliance, we would write 
    # fhmm.train(train_elec, sample_period=60)
    clf.train(tran_elec_all, sample_period=sample_period)
    end = time.time()
    print("Runtime =", end-start, "seconds.")
    gt, predictions[clf_name] = predict(clf, test_elec, sample_period, train.metadata['timezone'])

appliance_labels = [m.label() for m in gt.columns.values]
gt.columns = appliance_labels
predictions['CO'].columns = appliance_labels
predictions['FHMM'].columns = appliance_labels
pprint(gt.head())
pprint(predictions['CO'].head())
pprint(predictions['FHMM'].head())

predictions['CO']['Fridge'].head(6000).plot(label="Pred")
gt['Fridge'].head(6000).plot(label="GT")
plt.pyplot.legend()
plt.pyplot.show()

predictions['FHMM']['Fridge'].head(6000).plot(label="Pred")
gt['Fridge'].head(6000).plot(label="GT")
plt.pyplot.legend()
plt.pyplot.show()

rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = nilmtk.utils.compute_rmse(gt, predictions[clf_name])

rmse = pd.DataFrame(rmse)
pprint(rmse)

print("GT time range:", gt.index.min(), "to", gt.index.max())
print("Predictions time range (CO):", predictions['CO'].index.min(), "to", predictions['CO'].index.max())
print("Predictions time range (FHMM):", predictions['FHMM'].index.min(), "to", predictions['FHMM'].index.max())