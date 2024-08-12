# A simple script to do round robin random sampling of iNaturalist observations starting from lowest count species and iteratively increasing unitl a desired number of samples is reached.
from collections import Counter
import pandas as pd
import os

LIMIT = 120000
thr = 100
df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/iNat/iNaT_metadata.csv")

#select species with counts exceeding threshold of thr
species_count =  dict(Counter(df.scientific_name))
selected_species_list = [(k,v) for k,v in species_count.items() if v >= thr]
selected_species =[k for k,v in selected_species_list]
selected_species_dict = dict(sorted(selected_species_list, key=lambda x: x[1]))

selected_species_sample_dict = dict([(k,thr) for k,v in selected_species_list])
#initialize total count
count = len(selected_species)*thr

while count < LIMIT:
    for s in selected_species:
        if selected_species_sample_dict[s] < selected_species_dict[s]:
            selected_species_sample_dict[s] += 1
            count += 1
        else:
            continue


ids_to_sample = []
for s in selected_species:
    df_s = df[df['scientific_name']==s]
    df_s = df_s.sample(selected_species_sample_dict[s],random_state=42)
    ids_to_sample = ids_to_sample + list(df_s.id)

df_final = df[df['id'].isin(ids_to_sample)]
print(df_final.shape) #(120016, 41)
print(len(set(df_final.scientific_name))) #611 species
print(Counter(df_final.scientific_name))

df_final.to_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/iNat/iNaT_metadata_species_balanced.csv")
