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
        # key = get_key(file.rstrip(".csv"))
        # print(f"Processing file: {file}, key: {key}")
        df = pd.read_csv(csv_path, usecols=['Time', 'P1', 'Q1'], dayfirst=True)
        df.set_index('Time', inplace=True)
        df = df.sort_index()
        # df = pd.read_csv(
        #     csv_path,
        #     parse_dates=[0],
        #     index_col=0,
        #     skipinitialspace=True,
        #     delimiter=",",
        #     usecols=['Time', 'P1', 'P2'],
        #     header=0,
        #     dayfirst=True
        #     )
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S") # Konvertierung hiermit viel länger

        duplicates = df.index.duplicated()
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicate timestamps.")
            print(df[duplicates])
            df = df[~duplicates]
            print("Duplicates removed")

        print("Dataframe")
        print(df.head())

        for col in df.columns:
            print(f"Column: {col}")
            key = get_key()
            print(f"Key: {key}")
            chan_df = df[[col]].copy()    # = pd.DataFrame(df[col])
            chan_df.columns = pd.MultiIndex.from_tuples([("power", "active")])
            chan_df.columns.set_names(["measurement", "type"], inplace=True)

            print(f"DataFrame shape: {df.shape}")
            print(chan_df.head())
            print(f"Columns before storing in HDF5: {df.columns}")
            sys.stdout.flush()
            datastore.put(key, chan_df)

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