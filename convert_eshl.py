import pandas as pd
import numpy as np
from nilmtk.utils import get_datastore, check_directory_exists, get_module_directory
import os
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore
from pprint import pprint

def convert_eshl(input_path, output_filename, format="HDF"):
    check_directory_exists(input_path)
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and '.csv' in f]
    # file.csv in Liste
    files.sort()
    datastore = get_datastore(output_filename, format, mode='w')    # HDFDataStore object
    
    for i, file in enumerate(files):
        csv_path = os.path.join(input_path, files[i])
        key = get_key(file.rstrip(".csv"))
        print(f"Processing file: {file}, key: {key}")
        df = pd.read_csv(
            csv_path, parse_dates=[0], 
            index_col=0, skipinitialspace=True, 
            delimiter=",", 
            names=["Time", "P1", "P2", "P3", "Q1", "Q2", "Q3"],
            header=0
            )
        
        print(f"DataFrame shape: {df.shape}")
        print(df.head())

        #df.set_index(df.columns[0], inplace=True)
        print(f"Columns before storing in HDF5: {df.columns}")
        datastore.put(key, df[['P1', 'P2', 'P3', 'Q1', 'Q2', 'Q3']])

    datastore.close()


    print('Processing metadata...')
    metadata_path = os.path.join(get_module_directory(), 'dataset_converters', 'eshl', 'metadata')
    convert_yaml_to_hdf5(metadata_path, output_filename)

# def get_key(string):
#     split = string.split("_")
#     assert split[0].startswith("METER") or split[0].startswith("WIZ")
#     meter = split[0]
#     clamp_number = split[1].replace("Clamp", "")
#     return meter + "/" + "Clamp" + clamp_number
def get_key(string):
    if not hasattr(get_key, "counter"):
        get_key.counter = 1  # Initialize the counter on the first call
    key = "building1/elec/meter" + str(get_key.counter)
    get_key.counter += 1  # Increment the counter
    return key

if __name__ == "__main__":
    input_path = "C:/Users/Megapoort/Desktop/nilmdata/eshl"
    output_filename = "C:/Users/Megapoort/Desktop/nilmdata/eshl/eshl.h5"
    convert_eshl(input_path, output_filename)