#This script performs reverse geo-coding to find address for all the locations on our dataset.

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.point import Point
from cleantext import clean
import os
from tqdm import tqdm

geolocator = Nominatim(user_agent="openmapquest")
def reverse_geocoding(lat, lon):
    try:
        location = geolocator.reverse(Point(lat, lon),language='en')
        address = location.address
        return address
    except:
        address = "Not Found"
        return address
    
data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound"
meta_df = pd.read_csv(os.path.join(data_path,"source_balanced_filtered_geotagged_sounds.csv"),low_memory=False)

addresses = []
keys = []
sources = []
lats  = []
longs = []

for i in tqdm(range(len(meta_df))):
    row = meta_df.iloc[i]
    lat = row.latitude
    long = row.longitude
    source = row.source
    key = row.key
    address = reverse_geocoding(lat=row.latitude, lon=row.longitude)
    print('The Address for {} from {} with lat, lon: {},{} is:{}\n'.format(row.key,row.source,row.latitude, row.longitude, address))
    keys.append(key)
    sources.append(source)
    lats.append(lat)
    longs.append(long)
    addresses.append(address)
    # import code;code.interact(local=dict(globals(), **locals()));

address_df = pd.DataFrame(columns=['key','source','latitude','longitude','address'])
address_df['key'] = keys
address_df['source'] = sources
address_df['latitude'] = lats
address_df['longitude'] = longs
address_df['address'] = addresses

address_df.to_csv(os.path.join(data_path,"source_balanced_filtered_geotagged_sounds_address.csv"))