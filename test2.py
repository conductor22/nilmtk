import pandas as pd
import numpy as np
from nilmtk.utils import get_datastore, check_directory_exists, get_module_directory
import os
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore
from nilmtk.measurement import LEVEL_NAMES
from pprint import pprint
import sys

def convert_eshl(input_path, output_filename, format="HDF"):
    check_directory_exists(input_path)
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and '.csv' in f]
    # file.csv in Liste
    files.sort()
    datastore = get_datastore(output_filename, format, mode='w')    # HDFDataStore object
    
    for i, file in enumerate(files):
        csv_path = os.path.join(input_path, files[i])
        usecols=['Timestamp','mains',
                 'television','fan','fridge',
                 'laptop computer','electric heating element',
                 'oven','unknown','washing machine',
                 'microwave','toaster',
                 'sockets','cooker'
                ]
        df = pd.read_csv(csv_path, usecols, 3, )

        if sort_index:
            df = df.sort_index() # might not be sorted...
        chan_id = 0
        for col in df.columns:
            chan_id += 1
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=nilmtk_house_id, meter=chan_id)
            
            chan_df = pd.DataFrame(df[col])
            chan_df.columns = pd.MultiIndex.from_tuples([('power', 'apparent')])
            
            # Modify the column labels to reflect the power measurements recorded.
            chan_df.columns.set_names(LEVEL_NAMES, inplace=True)
            store.put(str(key), chan_df)
        df = pd.read_csv(csv_path, usecols=['Time', 'P1'], dayfirst=True)
        df.set_index('Time', inplace=True)
        df = df.sort_index()
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S") # Konvertierung hiermit viel länger

        duplicates = df.index.duplicated()
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicate timestamps.")
            print(df[duplicates])

        key = get_key()

        df.columns = pd.MultiIndex.from_tuples([("power", "phase1")])   # , ("power", "phase2")
        df.columns.set_names(["measurement", "type"], inplace=True)

        print("Prepared Dataframe:")
        print(df)
        datastore.put(key, df)
        # for col in df.columns:
        #     print(f"Column: {col}")
        #     # key = get_key()
        #     print(f"Key: {key}")
        #     chan_df = pd.DataFrame(df[col])
        #     chan_df.columns = pd.MultiIndex.from_tuples([("power", "phase1")])
        #     chan_df.columns.set_names(["measurement", "type"], inplace=True)

        #     print(f"DataFrame shape: {df.shape}")
        #     print(chan_df.head())
        #     print(f"Columns before storing in HDF5: {df.columns}")
        #     sys.stdout.flush()
        #     datastore.put(key, chan_df)

    # datastore.close()


    print('Processing metadata...')
    metadata_path = os.path.join(get_module_directory(), 'dataset_converters', 'eshl', 'metadata')
    convert_yaml_to_hdf5(metadata_path, output_filename)
    datastore.close()


def get_key():
    if not hasattr(get_key, "counter"):
        get_key.counter = 1
    key = "building1/elec/meter" + str(get_key.counter)
    get_key.counter += 1
    return key

if __name__ == "__main__":
    input_path = "C:/Users/Megapoort/Desktop/nilmdata/eshl"
    output_filename = "C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5"
    convert_eshl(input_path, output_filename)