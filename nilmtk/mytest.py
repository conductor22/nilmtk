from nilmtk import DataSet
from nilmtk.utils import print_dict

ds = DataSet("E:/Users/Megapoort/nilmtk/nilmtk/data/random.h5")
print("*****************")

import json
print(json.dumps(ds.metadata, indent=4))
print("*****************")

print(ds.buildings[1].elec)
print("*****************")


from nilmtk.disaggregate import combinatorial_optimisation
disaggregator = combinatorial_optimisation()
disaggregator.train(ds)
disaggregated_data = disaggregator.disaggregate(ds.buildings[1])
print(disaggregated_data)