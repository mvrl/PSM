import requests
import os
from config import cfg
import pandas as pd
from collections import Counter
import argparse
from tqdm import tqdm
import time

LIMIT = 14000 #Limit of downloads per job
audio_path = os.path.join(cfg.iNat_data_path,"raw_audio")
metadata_path = cfg.iNat_metadata_path

## Using the following function, simple cleaning of metadata was already done.
def clean_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=['id', 'sound_url', 'latitude', 'longitude'])
    urls = list(df['sound_url'])
    formats = [u.split('/')[-1].split('?')[0].split('.')[1] for u in urls]
    df['sound_format'] = formats
    ##Counter(formats)
    format_count = {'m4a': 212030, 'wav': 150624, 'mp3': 85350, 'mpga': 4085} #top count of audio formats>500
    df = df[df['sound_format'].isin(format_count.keys())]
    df.to_csv(metadata_path)


def download_sound(url, filename):
    if not os.path.exists(filename):
        try:
            time.sleep(1)
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                return True
            else:
                print('Download failed for {}'.format(filename,url))
                return False
        except:
            print('Download failed for {}'.format(filename,url))
            return False
    else:
        return True


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download sounds from iNaturalist')
parser.add_argument('--csv_file', type=str, default = metadata_path,
                    help='Path to the CSV file containing sound IDs')
parser.add_argument('--split_id', type=str,
                    help='Split ID for blocks of $LIMIT sound IDs')
parser.add_argument('--output_directory', type=str, default = audio_path,
                    help='Directory to save the downloaded sounds')
args = parser.parse_args()


# Read sound IDs from the CSV file
sound_ids = list(pd.read_csv(args.csv_file)['id'])
sound_urls = list(pd.read_csv(args.csv_file)['sound_url'])
sound_formats = list(pd.read_csv(args.csv_file)['sound_format'])

meta_df = pd.read_csv(args.csv_file)[['id','sound_url','sound_format']]

# Split sound IDs into blocks of $LIMIT
split_id = int(args.split_id)
start_index = (split_id - 1) * LIMIT
end_index = split_id * LIMIT

sub_df = meta_df.iloc[start_index:end_index]


for i in tqdm(range(len(sub_df)), desc='Downloading for split:'+str(split_id)):
    row = sub_df.iloc[i]
    sound_id = row.id
    url = row.sound_url
    sound_format = row.sound_format
    filename = os.path.join(audio_path,str(sound_id)+'.'+sound_format)
    result = download_sound(url,filename)
    # import code;code.interact(local=dict(globals(), **locals()));
print('Download complete!')

##Usage: 
##python iNaturalist_download.py --csv_file /path/to/iNaT_metadata.csv --split_id n --output_directory /path/to/download