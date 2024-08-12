#Merging geolocations from different sources. The merged csv will be used for overhead imagery download

import os
import pandas as pd
from collections import Counter
from config import cfg

def get_merged_df():
    freesound_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/freesound/freesound_successful_downloads.csv")
    freesound_geotags =  list(freesound_df.geotag)
    freesound_lats =  [str(g).split(" ")[0] for g in freesound_geotags]
    freesound_lons =  [str(g).split(" ")[1] for g in freesound_geotags]
    freesound_df['latitude'] = freesound_lats
    freesound_df['longitude'] = freesound_lons
    
    aporee_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/final_metadata_with_captions.csv") 
    iNat_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/iNat/iNaT_metadata_species_balanced.csv")                     
    
    yfcc_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/yfcc/yfcc100m_metadata.csv")                  
    done_keys = [i.split(".")[0] for i in os.listdir("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/yfcc/raw_audio")]
    yfcc_df = yfcc_df[yfcc_df['key'].isin(done_keys)]

    key_cols = {'freesound':'id','aporee':'long_key','iNat':'id','yfcc':'key'}
    merge_df = pd.DataFrame(columns=['key','source','latitude','longitude'])

    merge_df['key'] = list(freesound_df[key_cols['freesound']]) + list(aporee_df[key_cols['aporee']]) + list(iNat_df[key_cols['iNat']]) + list(yfcc_df[key_cols['yfcc']])
    merge_df['source']  = ['freesound']*len(freesound_df) + ['aporee']*len(aporee_df) + ['iNat']*len(iNat_df) + ['yfcc']*len(yfcc_df)
    merge_df['latitude']= list(freesound_df['latitude']) + list(aporee_df['latitude']) + list(iNat_df['latitude']) + list(yfcc_df['latitude'])
    merge_df['longitude'] = list(freesound_df['longitude']) + list(aporee_df['longitude']) + list(iNat_df['longitude']) + list(yfcc_df['longitude'])

    return merge_df

if __name__ == "__main__": 
    data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound"
    meta_df = get_merged_df()
    print(Counter(meta_df.source)) #Counter({'iNat': 120016, 'yfcc': 98506, 'aporee': 50792, 'freesound': 49677})
    print(meta_df.shape) #(318991, 4)
    meta_df.to_csv(os.path.join(data_path,"source_balanced_geotagged_sounds.csv"))
    print("Merged csv file saved for all data sources")