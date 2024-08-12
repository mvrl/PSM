# Some of the images might be corrupt. In an attempt to finding those files, this script lists the filesize of all images downloaded from bingmap.

import os
import pandas as pd
from tqdm import tqdm
from collections import Counter

data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw"
meta_df = pd.read_csv(os.path.join(data_path,"/metafiles/GeoSound/source_balanced_geotagged_sounds.csv"))
meta_df = meta_df.dropna()
sizes1 = []
sizes2 = []
for i in tqdm(range(len(meta_df))):
    row = meta_df.iloc[i]
    key = row.key
    source = row.source
    out_file1 = f'{data_path}/{source}/{"images/bingmap"}/{key}.jpeg'
    out_file2 = f'{data_path}/{source}/{"images/sentinel"}/{key}.jpeg'
    file_size1 = os.path.getsize(out_file1)/1024
    file_size2 = os.path.getsize(out_file2)/1024
    sizes1.append(file_size1)
    sizes2.append(file_size2)
meta_df['bingmap_image_size'] = sizes1
meta_df['sentinel_image_size'] = sizes2

#filtering out the files that are likely to be problematic:
meta_df = meta_df[(meta_df['sentinel_image_size']>25)&(meta_df['bingmap_image_size']>50)] # 318653 sounds
meta_df.to_csv(os.path.join(data_path,"/metafiles/GeoSound/source_balanced_filtered_geotagged_sounds.csv"))
print(Counter(meta_df.source)) #Counter({'iNat': 119899, 'yfcc': 98236, 'aporee': 50784, 'freesound': 49734})
print(meta_df.shape)

