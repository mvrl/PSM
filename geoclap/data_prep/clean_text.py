#This script is responsible for basic text cleaning after merging metadata from all four sources
import pandas as pd
import os
from cleantext import clean
import re
from tqdm import tqdm

def clean_description(text):
    sent = re.sub(r'(<br\s*/>)',' ',text)
    output = clean(sent,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks= True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                   # replace all URLs with a special token
        no_emails=True,                 # replace all email addresses with a special token
        no_phone_numbers=True,          # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=True,       # replace all currency symbols with a special token   
        no_punct= True, 
        replace_with_punct=" ",  
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling. Besides the default English ('en'), only German ('de') is supported
                )

    output = re.sub(r'\s+',' ',output)
    allow = ["<",">"," "]
    output  = "".join([ c if (c.isalnum() or c in allow) else "" for c in output])
    return output

if __name__ == "__main__": 
    data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw"
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
    print(freesound_df.shape, aporee_df.shape, iNat_df.shape, yfcc_df.shape)

    merged_meta_df = pd.concat([freesound_df,aporee_df,iNat_df,yfcc_df]).fillna('')
    
    texts = []
    for i in tqdm(range(len(merged_meta_df))):
        row = merged_meta_df.iloc[i]
        if row.source == 'freesound':
            text = ' '.join([row.description, row.tags])
        elif row.source == 'aporee':
            text = ' '.join([row.description, row.title])
        elif row.source == 'iNat':
            text = ' '.join([row.description, row.common_name])
        elif row.source == 'yfcc':
            text = ' '.join([row.description, row.tags ,row.title])
            text =  text.replace("+",' ')

        processed_text  = clean_description(text)
        #import code;code.interact(local=dict(globals(), **locals()));
        texts.append(processed_text)
    
    merged_meta_df['text'] = texts
    merged_meta_df.to_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/merged_metadata_with_text.csv")
    
    