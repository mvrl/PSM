import csv
import gzip
import os
import json
from tqdm import tqdm
from config import cfg

metadata_columns = ['photoid', 'uid', 'unickname', 'datetaken', 'dateuploaded', 'capturedevice', 'title',
                    'description', 'usertags', 'machinetags', 'longitude', 'latitude', 'accuracy', 'pageurl',
                    'downloadurl', 'licensename', 'licenseurl', 'serverid', 'farmid', 'secret', 'secretoriginal',
                    'ext', 'marker','key']

## Note: Each video sample has name as follows: key.mp4", where key has following format: "aaabbbXXXXX...."
## In each key, first three characters (aaa) indicate shard id and next three characters (bbb) indicate sub directory within the shard.

# Step 2 and 3: Iterate through .gz files and extract values
directory = cfg.yfcc_metadata_path
with open(os.path.join(cfg.yfcc_data_path,"yfcc100m_metadata.csv"), "w", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Step 4: Write header row based on the column names
    csvwriter.writerow(metadata_columns)

    # Step 2 and 3 (Continued): Iterate through .gz files and extract values
    gz_files = [file for file in os.listdir(directory) if file.endswith('.gz')]
    for gz_file in tqdm(gz_files, desc="Processing Files"):
        with gzip.open(os.path.join(directory, gz_file), 'rt', encoding='utf-8') as f:
            for line in f:
                # Parse the line as JSON
                data = json.loads(line)

                longitude = data.get('longitude')
                latitude = data.get('latitude')
                marker = data.get('marker')
                # import code;code.interact(local=dict(globals(), **locals()));
                if (marker == 1) and longitude is not None and latitude is not None: #Select only geotagged videos
                    # Select only the desired columns based on metadata_columns
                    data_dict = {col: data.get(col) for col in metadata_columns}

                    # Step 5: Write data_dict values to the CSV file
                    csvwriter.writerow(data_dict.values())

print("All .gz files processed. Data written to yfcc100m_metadata.csv.")
