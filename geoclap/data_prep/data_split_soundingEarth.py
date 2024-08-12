#This is similar to data_split.py, the only differences being it works only on the Sounding earth dataset.
#Therefore this script does not require source balanced sampling procedure carried out for GeoSound dataset.

# This script performs train/test/split using following strategy:
# 1. First divide the world into 10km x 10km  cells.
# 2. Select the cells that have at least 25 observations.
# 3. Group these cells into three categories: high, medium, low based on 0.33 and 0.66 quantile of number of observations in them.
# 4. For each category, randomly select 35% of cells to be held out for non-train split.
# 5. From the held out cells, randomly sample 40% to validation and the rest to test split.


import pandas as pd
import os
import numpy as np
from collections import Counter
import random
import math

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

bin_thr = 25                                                    # Threshold to determine if the cells are to be included for train/val/test split algorithm
nontrainfrac = 0.35                                             # fraction of cells to be held for test and val split.
data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/"
train_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/train_df.csv")
val_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/validate_df.csv")
test_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/test_df.csv")
original_meta_df = pd.concat([train_df,val_df, test_df])
original_meta_df = original_meta_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

sample_ids = [str(original_meta_df.iloc[i]['long_key']) for i in range(len(original_meta_df))]
original_meta_df['sample_id'] = sample_ids

meta_df = original_meta_df

# Function to calculate bin ID
def calculate_bin_id(row):
    # Convert latitude and longitude to kilometers
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * math.cos(math.radians(row['latitude']))

    # Cell size for 10km by 10km
    cell_size_km = 10.0

    # Calculate the number of degrees equivalent to 10km
    degrees_per_cell_lat = cell_size_km / km_per_degree_lat
    degrees_per_cell_lon = cell_size_km / km_per_degree_lon

    # Calculate latitude and longitude bin indices
    latitude_bin = int((row['latitude'] + 90) // degrees_per_cell_lat)
    longitude_bin = int((row['longitude'] + 180) // degrees_per_cell_lon)

    # Calculate the bin ID
    bin_id = latitude_bin * 360 + longitude_bin
    return bin_id


# Function to split bin ids to train/val/test
def split_bin_ids(hist):
    M = len(hist) 
    valtest_bins = np.random.choice(hist['bin_id'],int(nontrainfrac*M),replace = False)
    train_bins = [b for b in hist['bin_id'] if b not in valtest_bins]
    val_bins = list(np.random.choice(valtest_bins,int(0.40*len(valtest_bins)),replace = False))
    test_bins = [b for b in valtest_bins if b not in val_bins]
    bins = {'train':train_bins,'val':val_bins, 'test':test_bins}
    return bins

bin_ids = []
for i in range(len(meta_df)):
    row = meta_df.iloc[i]
    bin_ids.append(calculate_bin_id(row))

meta_df['bin_id'] = bin_ids
all_ids = list(set(bin_ids))
print("Total number of data containing bins:",len(all_ids))

c = dict(Counter(meta_df['bin_id']))
hist_df = pd.DataFrame(c.items(), columns=['bin_id', 'count'])
# import code;code.interact(local=dict(globals(), **locals()));
selected_hist= hist_df[hist_df['count']>=bin_thr]
print("Total number of selected bins:",len(selected_hist))

q1 = selected_hist['count'].quantile(0.33)
q2 = selected_hist['count'].quantile(0.66)

low_hist = selected_hist[selected_hist['count']<=q1]
medium_hist = selected_hist[(selected_hist['count']>q1)&(selected_hist['count']<=q2)]
high_hist = selected_hist[selected_hist['count']>q2]

low_bins = split_bin_ids(low_hist)
medium_bins = split_bin_ids(medium_hist)
high_bins = split_bin_ids(high_hist)

left_bins = [b for b in all_ids if b not in list(selected_hist['bin_id'])]
TRAIN_bins = low_bins['train'] + medium_bins['train'] + high_bins['train'] + left_bins
VAL_bins = low_bins['val'] + medium_bins['val'] + high_bins['val']
TEST_bins = low_bins['test'] + medium_bins['test'] + high_bins['test']

#sanity check:
print("SANITY CHECK BASED ON BIN IDs. Should return empty")
print(set(TRAIN_bins).intersection(VAL_bins))
print(set(TRAIN_bins).intersection(TEST_bins))
print(set(TEST_bins).intersection(VAL_bins))
print("number of bins in train/val/test split:")
print(len(TRAIN_bins), len(VAL_bins), len(TEST_bins))

train_df = meta_df[meta_df['bin_id'].isin(TRAIN_bins)]
val_df = meta_df[meta_df['bin_id'].isin(VAL_bins)]
test_df = meta_df[meta_df['bin_id'].isin(TEST_bins)]

train_sample_id = [str(train_df.iloc[i].long_key) for i in range(len(train_df))]
train_df['sample_ID'] = train_sample_id

val_sample_id = [str(val_df.iloc[i].long_key) for i in range(len(val_df))]
val_df['sample_ID'] = val_sample_id

test_sample_id = [str(test_df.iloc[i].long_key) for i in range(len(test_df))]
test_df['sample_ID'] = test_sample_id
print("Overall sample split:")

print(len(train_df), len(val_df), len(test_df))
print("TRAIN/VAL/TEST split done!")
print("FINAL SANITY CHECK ON SAMPLE_ID. Should return empty")
print(set(train_df['sample_ID']).intersection(val_df['sample_ID']))
print(set(train_df['sample_ID']).intersection(test_df['sample_ID']))
print(set(val_df['sample_ID']).intersection(test_df['sample_ID']))

# exec(os.environ.get('DEBUG'))
train_df.to_csv(os.path.join(data_path,"aporee_train_fairsplit_10km.csv"))
val_df.to_csv(os.path.join(data_path,"aporee_val_fairsplit_10km.csv"))
test_df.to_csv(os.path.join(data_path,"aporee_test_fairsplit_10km.csv"))

# number of bins in train/val/test split:
# 6128 49 77
# Overall sample split:
# 41469 3269 6054  #evaluation {41469 3242 5801}
# TRAIN/VAL/TEST split done!
# FINAL SANITY CHECK ON SAMPLE_ID. Should return empty
# set()
# set()
# set()

# Actual test gallery size: 5801
# Actual val gallery size: 3242