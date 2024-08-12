#This script does a final sanity check, adds address as an additional column of metadata and prepares a merged_metadata_final.csv to be used for our experiments.
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter

def get_metafiles(data_path):
    freesound_df = pd.read_csv(os.path.join(data_path,"freesound/freesound_successful_downloads.csv")).loc[:,['id', 'date', 'latitude', 'longitude', 'description', 'tags']]                                                                                                            
    aporee_df = pd.read_csv(os.path.join(data_path,"aporee/final_metadata_with_captions.csv")).loc[:,['long_key', 'latitude', 'longitude', 'description', 'title','date_recorded']] 
    iNat_df = pd.read_csv(os.path.join(data_path,"iNat/iNaT_metadata_species_balanced.csv")).loc[:,['id', 'latitude', 'longitude', 'description', 'scientific_name', 'common_name', 'sound_format','observed_on_string']]                     
    yfcc_df = pd.read_csv(os.path.join(data_path,"yfcc/yfcc100m_metadata.csv")).loc[:,['key', 'latitude', 'longitude', 'description','title', 'usertags','datetaken']]  
    final_geolocations = pd.read_csv(os.path.join(data_path,"source_balanced_filtered_geotagged_sounds.csv"))

    freesound_df = freesound_df[freesound_df['id'].astype('str').isin(list(final_geolocations[final_geolocations['source']=='freesound']['key'].astype('str')))].rename(columns={'id':'key'})
    freesound_df['source'] = ['freesound']*len(freesound_df)

    aporee_df = aporee_df[aporee_df['long_key'].astype('str').isin(list(final_geolocations[final_geolocations['source']=='aporee']['key'].astype('str')))].rename(columns={'long_key':'key','date_recorded':'date'})
    aporee_df['source'] = ['aporee']*len(aporee_df)

    iNat_df = iNat_df[iNat_df['id'].astype('str').isin(list(final_geolocations[final_geolocations['source']=='iNat']['key'].astype('str')))].rename(columns={'id':'key','tag_list':'tags','observed_on_string':'date'})
    iNat_df['source'] = ['iNat']*len(iNat_df)

    yfcc_df = yfcc_df[yfcc_df['key'].astype('str').isin(list(final_geolocations[final_geolocations['source']=='yfcc']['key'].astype('str')))].rename(columns={'usertags':'tags','datetaken':'date'})
    yfcc_df['source'] = ['yfcc']*len(yfcc_df)

    merged_meta = pd.concat([freesound_df,aporee_df,iNat_df,yfcc_df]).fillna('')

    return merged_meta


if __name__ == "__main__": 
    data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw"
    merged_meta = get_metafiles(data_path)
    meta_keys = list(merged_meta['key'].astype(str))
    meta_source = list(merged_meta['source'].astype(str))
    meta_sid = ["-".join([meta_source[i],meta_keys[i]]) for i in range(len(meta_keys))]
    merged_meta['sample_id'] = meta_sid

    merged_text_meta_df = pd.read_csv(os.path.join(data_path,"merged_metadata_with_text.csv"))

    merge_keys = list(merged_text_meta_df['key'].astype(str))
    merge_source = list(merged_text_meta_df['source'].astype(str))
    merge_sid = ["-".join([merge_source[i],merge_keys[i]]) for i in range(len(merge_keys))]
    merged_text_meta_df['sample_id'] = merge_sid

    final_addresses_df = pd.read_csv(os.path.join(data_path,"source_balanced_filtered_geotagged_sounds_address.csv"))
    final_addresses_df['key'] = final_addresses_df['key'].astype(str)
    address_keys = list(final_addresses_df['key'].astype(str))
    address_source = list(final_addresses_df['source'].astype(str))
    address_sid = ["-".join([address_source[i],address_keys[i]]) for i in range(len(address_keys))]
    final_addresses_df['sample_id'] = address_sid

    merged_meta = merged_meta[merged_meta['sample_id'].isin(address_sid)]
    merged_text_meta_df = merged_text_meta_df[merged_text_meta_df['sample_id'].isin(address_sid)] #get only the samples for which we were able to do reverse geocoding
    final_addresses_df = final_addresses_df[final_addresses_df['sample_id'].isin(address_sid)]
    

    merged_meta = merged_meta.sort_values(by=['sample_id'])
    merged_text_meta_df = merged_text_meta_df.sort_values(by=['sample_id'])
    final_addresses_df = final_addresses_df.sort_values(by=['sample_id'])

    print("sample order check:")
    print(list(merged_meta['sample_id']) == list(merged_text_meta_df['sample_id']))
    print(list(merged_meta['sample_id']) == list(final_addresses_df['sample_id']))

    merged_meta['text'] = list(merged_text_meta_df['text'])
    merged_meta['address'] = list(final_addresses_df['address'])
    untitled_columns = [col for col in merged_meta.columns if col.startswith('Unnamed')]
    df = merged_meta.drop(columns=untitled_columns)
    df.to_csv(os.path.join(data_path,"merged_metadata_final.csv"))
    print(df.shape)
    print(Counter(df['source']))
    