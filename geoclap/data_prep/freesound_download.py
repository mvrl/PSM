#Source: ChatGPT
import time
import argparse
import csv
import os
import freesound
from tqdm import tqdm
import pandas as pd
from config import cfg

LIMIT = 1995

def initialize_client(api_key):
    # Initialize the Freesound client
    freesound_client = freesound.FreesoundClient()
    freesound_client.set_token(api_key)

    return freesound_client

def download_sound(freesound_client,date,sound_id,sound_directory):
    entry = [sound_id,date]
    try: #To check if clent.get_sound does not fail
        sound = freesound_client.get_sound(sound_id,normalized=1)         
        entry = entry + [sound.url,sound.name," ".join(sound.tags),sound.geotag,sound.license,
                 sound.type,sound.channels,sound.bitrate,sound.bitdepth,sound.duration,sound.samplerate,
                 sound.username,sound.download,sound.description]
        #fields = ['preview_exists','id','url','name','tags','geotag','license',
        #  'type','channels','bitrate','bitdepth','duration','original_samplerate',
        #   'username','download','description']
        
        try: #To check if sound.retrieve_preview fails or not
            sound.retrieve_preview(sound_directory, name=str(sound.id)+".mp3")
            preview_exists = True
            success = True
            
        except: #To check if sound.retrieve fails or not
            preview_exists = False  
            sound.retrieve(sound_directory, name=str(sound.id)+sound.type)
            success = True     
                
        entry = [preview_exists] + entry
        
    except: #clent.get_sound fails
        success = False
    
    return entry, success

# Function to save the results to CSV files
def save_results_to_csv(fields, results, output_directory, split_id):
    successful_downloads_file = os.path.join(output_directory, 'freesound_successful_downloads_split'+str(split_id)+'.csv')
    failed_downloads_file = os.path.join(output_directory, 'freesound_failed_downloads_split'+str(split_id)+'.csv')

    with open(successful_downloads_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows([entry for entry, success in results if success])

    with open(failed_downloads_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows([entry for entry, success in results if not success])

    print(f'Successful downloads saved to: {successful_downloads_file}')
    print(f'Failed downloads saved to: {failed_downloads_file}')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download sounds from Freesound.org')
parser.add_argument('--api_key', type=str,default=cfg.freesound_api,
                    help='Freesound api key')
parser.add_argument('--csv_file', type=str,
                    help='Path to the CSV file containing sound IDs')
parser.add_argument('--split_id', type=str,
                    help='Split ID for blocks of $LIMIT sound IDs')
parser.add_argument('--output_directory', type=str,
                    help='Directory to save the downloaded sounds')
args = parser.parse_args()

# Create the output directory if it doesn't exist
sound_directory = os.path.join(args.output_directory,"raw_audio")
if not os.path.exists(sound_directory):
    os.makedirs(sound_directory)

freesound_client = initialize_client(args.api_key)
# Read sound IDs from the CSV file
meta_df = pd.read_csv(args.csv_file)
sound_ids = list(meta_df['Sound_ID'])

# Split sound IDs into blocks of $LIMIT
split_id = int(args.split_id)
start_index = (split_id - 1) * LIMIT
end_index = split_id * LIMIT
sound_ids_block = sound_ids[start_index:end_index]
results = {}
# Save results to CSV files
fields = ['preview_exists','id','date','url','name','tags','geotag','license',
         'type','channels','bitrate','bitdepth','duration','original_samplerate',
          'username','download','description']

# Downloaded sounds
successful_downloads_file = os.path.join(args.output_directory, 'freesound_successful_downloads.csv')
if os.path.exists(successful_downloads_file):
     downloaded_sounds = list(pd.read_csv(successful_downloads_file)['id'])
     sound_ids_block = [s for s in sound_ids_block if s not in downloaded_sounds]

results = []
for sound_id in tqdm(sound_ids_block, desc='Downloading'):
        date = meta_df[meta_df['Sound_ID']==sound_id].item()
        result = download_sound(freesound_client,date,sound_id,sound_directory)
        results.append(result)
        time.sleep(1)  # Sleep for 1 seconds

save_results_to_csv(fields,results, args.output_directory, args.split_id)

##Usage: 
##python freesound_download.py --api_key xxx --csv_file /path/to/freesound_geotagged_IDs.csv --split_id n --output_directory /path/to/download
                                      