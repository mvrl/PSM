# For the samples that passed sanity test using the script `./data_sanity.py`
# This script performs train/test/split using following strategy:
# 1. First divide the world into 1 degree x 1 degree cells.
# 2. Select the cells that have at least 25 observations.
# 3. Group these cells into three categories: high, medium, low based on 0.33 and 0.66 quantile of number of observations in them.
# 4. For each category, randomly select 10% of cells to be held out for non-train split.
# 5. From the held out cells, randomly sample 40% to validation and the rest to test split.
# 6. For the val/test split, 5000/10000 samples were sampled matching the source distribution of the train split.

import pandas as pd
import os
import numpy as np
from collections import Counter
import random
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

bin_thr = 25                                                    # Threshold to determine if the cells are to be included for train/val/test split algorithm
nontrainfrac = 0.10                                             # fraction of cells to be held for test and val split.
data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound"

original_meta_df = pd.read_csv(os.path.join(data_path,"merged_metadata_final.csv"),low_memory=False).sample(frac=1, random_state=SEED).reset_index(drop=True)
sample_ids = [original_meta_df.iloc[i]['source']+'-'+str(original_meta_df.iloc[i]['key']) for i in range(len(original_meta_df))]
original_meta_df['sample_id'] = sample_ids

status_df = pd.read_csv(os.path.join(data_path,"dataset_sanity.csv"))
sample_ids = [status_df.iloc[i]['source']+'-'+str(status_df.iloc[i]['key']) for i in range(len(status_df))]
status_df['sample_id'] = sample_ids
success_df = status_df[status_df['status']==True]
sample_ids = list(success_df.sample_id)

meta_df = original_meta_df[original_meta_df['sample_id'].isin(sample_ids)]
sampling_rates = []
for i in tqdm(range(len(meta_df))):
    sample_id = meta_df.iloc[i]['sample_id']
    sampling_rates.append(success_df[success_df['sample_id']==sample_id]['original_sampling_rate'].item())
meta_df['original_sampling_rate'] = sampling_rates
meta_df.to_csv(os.path.join(data_path,"merged_metadata_final_with_sr.csv"))
meta_df = pd.read_csv(os.path.join(data_path,"merged_metadata_final_with_sr.csv"))
# Function to calculate bin ID
def calculate_bin_id(row):
    latitude_bin = int(row['latitude'] + 90)
    longitude_bin = int(row['longitude'] + 180)
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

train_sample_id = [train_df.iloc[i].source + '_'+ str(train_df.iloc[i].key) for i in range(len(train_df))]
train_df['sample_ID'] = train_sample_id

val_sample_id = [val_df.iloc[i].source + '_'+ str(val_df.iloc[i].key) for i in range(len(val_df))]
val_df['sample_ID'] = val_sample_id

test_sample_id = [test_df.iloc[i].source + '_'+ str(test_df.iloc[i].key) for i in range(len(test_df))]
test_df['sample_ID'] = test_sample_id
print("Overall sample split:")
print("BEFORE SAMPLING")
print("Train split source distribution:",Counter(train_df['source']))
print("Val split source distribution:",Counter(val_df['source']))
print("Test split source distribution:",Counter(test_df['source']))

train_df.to_csv(os.path.join(data_path,"train_metadata.csv"))

val_df.to_csv(os.path.join(data_path,"val_metadata_whole.csv"))
test_df.to_csv(os.path.join(data_path,"test_metadata_whole.csv"))

def source_stratified_sampling(train_df,df,N):
    count_dict = dict(Counter(train_df['source']))
    count_dict_updated = {'iNat': None, 'yfcc': None, 'aporee': None, 'freesound': None}
    total = sum([v for v in count_dict.values()])
    sources =  list(count_dict.keys())
    df_lists = []
    for source in sources:
        source_frac = int(count_dict[source]/total*N)
        df_source = df[df['source']==source]
        count_dict_updated[source] = min(source_frac, len(df_source)) # min here to prevent "ValueError: Cannot take a larger sample than population when 'replace=False'"
    
    total = sum([v for v in count_dict_updated.values()])
    if total<N: # To get split of size of N, if the source distribution does not add up to N, we allow a few samples from iNat.
        count_dict_updated['iNat'] = count_dict_updated['iNat'] + N-total
    print("final source distribution for the split:",count_dict_updated)
    
    for source in sources:
        source_frac = count_dict_updated[source]
        df_source = df[df['source']==source]
        df_source =  df_source.sample(n=source_frac,random_state=SEED)
        df_lists.append(df_source)
    final_df = pd.concat(df_lists)
    return final_df

val_df = source_stratified_sampling(train_df,val_df,5000)
val_df.to_csv(os.path.join(data_path,"val_metadata.csv"))
test_df = source_stratified_sampling(train_df,test_df,10000)
test_df.to_csv(os.path.join(data_path,"test_metadata.csv"))

print("AFTER SAMPLING")
print("Train split source distribution:",Counter(train_df['source']))
print("Val split source distribution:",Counter(val_df['source']))
print("Test split source distribution:",Counter(test_df['source']))

print(len(train_df), len(val_df), len(test_df))
print("TRAIN/VAL/TEST split done!")
print("FINAL SANITY CHECK ON SAMPLE_ID. Should return empty")
print(set(train_df['sample_ID']).intersection(val_df['sample_ID']))
print(set(train_df['sample_ID']).intersection(test_df['sample_ID']))
print(set(val_df['sample_ID']).intersection(test_df['sample_ID']))

######### OUTPUT OF THE SCRIPT #####################
# Total number of data containing bins: 5372
# Total number of selected bins: 1498
# SANITY CHECK BASED ON BIN IDs. Should return empty
# set()
# set()
# set()
# number of bins in train/val/test split:
# 5224 58 90
# Overall sample split:
# BEFORE SAMPLING
# Train split source distribution: Counter({'iNat': 108754, 'yfcc': 92055, 'aporee': 46893, 'freesound': 46318})
# Val split source distribution: Counter({'iNat': 4748, 'yfcc': 3316, 'freesound': 1622, 'aporee': 1194})
# Test split source distribution: Counter({'iNat': 5486, 'yfcc': 2832, 'aporee': 2697, 'freesound': 1633})
# final source distribution for the split: {'iNat': 1851, 'yfcc': 1565, 'aporee': 797, 'freesound': 787}
# final source distribution for the split: {'iNat': 3999, 'yfcc': 2832, 'aporee': 1594, 'freesound': 1575}
# AFTER SAMPLING
# Train split source distribution: Counter({'iNat': 108753, 'yfcc': 92055, 'aporee': 46893, 'freesound': 46318})
# Val split source distribution: Counter({'iNat': 1851, 'yfcc': 1565, 'aporee': 797, 'freesound': 787})
# Test split source distribution: Counter({'iNat': 3999, 'yfcc': 2832, 'aporee': 1594, 'freesound': 1575})
# 294019 5000 10000
# TRAIN/VAL/TEST split done!
# FINAL SANITY CHECK ON SAMPLE_ID. Should return empty
# set()
# set()
# set()

# Actual test gallery size: 9931
# Actual val gallery size: 4976