#This script searches for IDs of geotagged audios available in www.freesound.org
from config import cfg
import csv
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
csv_file_path = cfg.freesound_geotagged_IDs_csv_path

# Set the base URL for FreeSound's search results
base_url = "https://freesound.org/search/?q=&f=duration%3A%5B0+TO+%2A%5D+is_geotagged%3A1&w=&s=Date+added+%28newest+first%29&advanced=1&g=&page={}"

# Set the number of pages you want to scrape 
num_pages = 3318

# Create a list to store the extracted metadata
metadata_list = []

# Loop through each page and extract metadata
for page in tqdm(range(1, num_pages + 1)):
    url = base_url.format(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the sample listings on the page
    sample_list = soup.find_all('div', class_='sample_player_small')
    # Extract metadata from each sample
    for sample in sample_list:
        # import code;code.interact(local=dict(globals(), **locals()));
        sound_id = int(sample['id'])
        date = sample.find_all('span',class_='date')[0].getText()

        metadata_list.append([sound_id, date])
    print(f"At page {page}, samples count is: {len(metadata_list)}, done %:{round(page/num_pages*100,2)}", end='\r')
# Save the metadata to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Sound_ID','date'])
    writer.writerows(metadata_list)

print(f'Saved {len(metadata_list)} samples to {csv_file_path}.')
