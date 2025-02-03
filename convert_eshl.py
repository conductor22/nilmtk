import pandas as pd
import numpy as np
from nilmtk.utils import get_datastore, check_directory_exists, get_module_directory
import os
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore
from nilmtk.measurement import LEVEL_NAMES
from pprint import pprint
import sys
import time

def convert_eshl(input_path, output_filename, format="HDF"):
    start = time.time()
    check_directory_exists(input_path)
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and '.csv' in f]
    # file.csv in Liste
    files.sort()
    datastore = get_datastore(output_filename, format, mode='w')    # HDFDataStore object
    wiz_list = []

    for i, file in enumerate(files):
        csv_path = os.path.join(input_path, files[i])
        df = pd.read_csv(csv_path, usecols=['Time', 'P1', 'P2', 'P3', 'Q1', 'Q2', 'Q3'], dayfirst=True)
        # df = pd.read_csv(csv_path, usecols=['Time', 'P1'], dayfirst=True)
        df.set_index('Time', inplace=True)
        df = df.sort_index()
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S") # Konvertierung hiermit viel länger
        df = df.tz_localize('Europe/Berlin')

        duplicates = df.index.duplicated()
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicate timestamps.")
            print(df[duplicates])
            df = df[~duplicates]
            print("Duplicates removed")

        df['P_total'] = df['P1'] + df['P2'] + df['P3']
        df.drop(columns=['P1', 'P2', 'P3'], inplace=True)

        df['Q_total'] = df['Q1'] + df['Q2'] + df['Q3']
        df.drop(columns=['Q1', 'Q2', 'Q3'], inplace=True)
        
        print("Dataframe")
        print(df.head())

        print(csv_path)
        if "WIZ" in csv_path:
            print("found a wiz")
            wiz_list.append(df)
            continue
        put_in_datastore(datastore, df)
    
    create_sitemeter(datastore, wiz_list)


    print('Processing metadata...')
    metadata_path = os.path.join(get_module_directory(), 'dataset_converters', 'eshl', 'metadata')
    convert_yaml_to_hdf5(metadata_path, output_filename)
    datastore.close()

    end = time.time()
    total = end - start
    minutes = int(total/60)
    seconds = total%60
    print(f"Total time: {minutes}m{seconds:.0f}s")

def get_key():
    if not hasattr(get_key, "counter"):
        get_key.counter = 1
    key = "building1/elec/meter" + str(get_key.counter)
    get_key.counter += 1
    return key

# def put_in_datastore(datastore, df):
#     for col in df.columns:
#             print(f"Column: {col}")
#             key = get_key()
#             print(f"Key: {key}")
#             chan_df = df[[col]].copy()    # = pd.DataFrame(df[col])
#             chan_df.columns = pd.MultiIndex.from_tuples([("power", "active")])
#             chan_df.columns.set_names(["physical_quantity", "type"], inplace=True)

#             print(f"DataFrame shape: {df.shape}")
#             print(chan_df.head())
#             print(f"Columns before storing in HDF5: {df.columns}")
#             sys.stdout.flush()
#             datastore.put(key, chan_df)
            

def put_in_datastore(datastore, df):
    key = get_key()
    print(f"Columns: {df.columns}")
    print(f"Key: {key}")

    if 'P_total' in df.columns and 'Q_total' in df.columns:
        chan_df = df[['P_total', 'Q_total']].copy()
        chan_df.columns = pd.MultiIndex.from_tuples([("power", "active"), ("power", "reactive")])
    else:
        raise ValueError("Couldn't find P_total and Q_total power columns")

    chan_df.columns.set_names(["physical_quantity", "type"], inplace=True)

    print(f"DataFrame shape: {chan_df.shape}")
    print(chan_df.head())
    print(f"Columns before storing in HDF5: {chan_df.columns}")
    sys.stdout.flush()
    datastore.put(key, chan_df)
        
def create_sitemeter(datastore, wiz_list):
    site_meter = wiz_list[0].copy()
    for i in range(1, len(wiz_list)):
        site_meter -= wiz_list[i]
    put_in_datastore(datastore, site_meter)

if __name__ == "__main__":
    input_path = "C:/Users/ieh-buergin/Desktop/eshl"
    output_filename = "C:/Users/ieh-buergin/Desktop/eshl/eshl.h5"
    convert_eshl(input_path, output_filename)