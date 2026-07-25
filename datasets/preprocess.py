import numpy as np

import matplotlib.pyplot as plt
import os
import h5py 
from tqdm import tqdm
from PIL import Image
import xarray as xr
import pandas as pd

def find_overlap_time(long_seq, long_dates, ds_ir108, ds_time):
    long_dates = long_dates.astype('datetime64[ns]')
    ds_time = ds_time.astype('datetime64[ns]')

    overlap_times, idx_long, idx_ds = np.intersect1d(
        long_dates,
        ds_time,
        return_indices=True
    )

    # print("Overlapping times:", overlap_times.shape)
    # print("Indices in long_dates:", idx_long.shape)
    # print("Indices in ds_time:", idx_ds.shape)

   
    # long_dates_filtered = long_dates[idx_long]
    # long_seq_filtered   = long_seq[idx_long, ...]

    # ds_time_filtered    = ds_time[idx_ds]
    # ds_ir108_filtered   = ds_ir108[idx_ds, ...]

    return idx_long, idx_ds



def calculate_dates(dates, start, end):
    start_date = dates[start]
    end_date = dates[end]
    start_np = np.datetime64(start_date)
    end_np   = np.datetime64(end_date)

    delta = end_np - start_np                # numpy.timedelta64
    diff_in_minutes = delta / np.timedelta64(1, 'm')
    return diff_in_minutes

THRESHOLD = 3.0
data_dir = ""

train_path = ""
test_path = ""
train_dirs = sorted(os.listdir(data_dir+"train/"))
test_dirs = sorted(os.listdir(data_dir+"test/"))

h5_data = ""

h5_file = h5py.File(h5_data, 'w')
h5_file.create_group('train')
h5_file.create_group('test')

seq_len = 0
total = 0
min_vals = 30000
min_key = ""

ds_2016 = None
ds_2017 = None
for dir in tqdm(train_dirs):
    img_paths = os.listdir(os.path.join(train_path, dir+"/"))
    long_seq = []
    long_dates = []

    for path in sorted(img_paths):
        img_path = os.path.join(train_path, dir, path)
        np_file = np.load(img_path, allow_pickle=True)
        data = np_file['data']
        dates = np_file['dates']
        long_seq.append(data)
        long_dates.append(dates)
        
    if "2016" in img_path:
        if ds_2016 == None:
            ds_2016 = xr.open_dataset("IR108_NW_2016.nc")
            ds_time = pd.to_datetime(ds_2016["time"].values)
            ds_ir108= ds_2016["IR108"].to_numpy()
    elif "2017" in img_path:
        if ds_2017 == None:
            ds_2017 = xr.open_dataset("IR108_NW_2017.nc")
            ds_time = pd.to_datetime(ds_2017["time"].values)
            ds_ir108= ds_2017["IR108"].to_numpy()

    long_seq = np.concatenate(long_seq, axis=0)
    long_dates = np.concatenate(long_dates, axis=0)
    print("vil:",long_seq.shape, long_dates.shape)
    print("ir108:",ds_ir108.shape, ds_time.shape)

    idx_long, idx_ds = find_overlap_time(long_seq, np.asarray(long_dates), ds_ir108, np.asarray(ds_time) )
    
    ds_time_fileter    = ds_time[idx_ds]
    ds_ir108_filtered   = ds_ir108[idx_ds, ...]
    
    total += long_dates.shape[0]
    i = 0

    print(long_seq.shape, long_dates.shape, ds_ir108_filtered.shape, ds_time_fileter.shape)

    for i, idx in enumerate(range(len(idx_long))):
        if long_seq[idx].mean() > THRESHOLD and calculate_dates(long_dates, idx-5, idx+20) == 5*25:        #THRESHOLD'
            seq = long_seq[idx-5:idx+20]
            seq[seq == 255] = 0
            all_means = sum([frame.mean() for frame in seq])
            # if all_means > 3.5 * 30:
            key = str(seq_len)
            if all_means < min_vals:
                min_vals = all_means
                min_key = key
            h5_file['train'].create_dataset(str(key), data=seq, dtype='uint8', compression='lzf')
            h5_file['train'].create_dataset(str(key)+"_ir108", data=ds_ir108_filtered[i], dtype='uint8', compression='lzf')
            seq_len += 1
            
h5_file['train'].create_dataset('all_len', data=seq_len)

print(total, seq_len)
print(min_vals)
print(min_key)

seq_len = 0
total = 0
min_vals = 30000
min_key = ""

ds_2018 = None
for dir in tqdm(test_dirs):
    img_paths = os.listdir(os.path.join(test_path,dir))
    long_dates = []
    long_seq = []

    for path in sorted(img_paths):
        img_path = os.path.join(test_path, dir, path)
        np_file = np.load(img_path, allow_pickle=True)
        data = np_file['data']
        dates = np_file['dates']
        long_seq.append(data)
        long_dates.append(dates)

    if "2018" in img_path:
        if ds_2018 == None:
            ds_2018 = xr.open_dataset("IR108_NW_2018.nc")
            ds_time = pd.to_datetime(ds_2018["time"].values)
            ds_ir108= ds_2018["IR108"].to_numpy()
    
    long_seq = np.concatenate(long_seq, axis=0)
    long_dates = np.concatenate(long_dates, axis=0)
    print("vil:",long_seq.shape, long_dates.shape)
    print("ir108:",ds_ir108.shape, ds_time.shape)

    idx_long, idx_ds = find_overlap_time(long_seq, np.asarray(long_dates), ds_ir108, np.asarray(ds_time) )
    
    ds_time_fileter    = ds_time[idx_ds]
    ds_ir108_filtered   = ds_ir108[idx_ds, ...]

    total += long_dates.shape[0]
    i = 0

    print(long_seq.shape, long_dates.shape, ds_ir108_filtered.shape, ds_time_fileter.shape)

    for i, idx in enumerate(range(len(idx_long))):
        if long_seq[idx].mean() > THRESHOLD and calculate_dates(long_dates, idx-5, idx+20) == 5*25:        #THRESHOLD'
            seq = long_seq[idx-5:idx+20]
            seq[seq == 255] = 0
            all_means = sum([frame.mean() for frame in seq])
            # if all_means > 3.5 * 30:
            key = str(seq_len)
            if all_means < min_vals:
                min_vals = all_means
                min_key = key
            h5_file['test'].create_dataset(str(key), data=seq, dtype='uint8', compression='lzf')
            h5_file['test'].create_dataset(str(key)+"_ir108", data=ds_ir108_filtered[i], dtype='uint8', compression='lzf')
            seq_len += 1

h5_file['test'].create_dataset('all_len', data=seq_len)

h5_file.close()
print(total, seq_len)
print(min_vals)
print(min_key)



