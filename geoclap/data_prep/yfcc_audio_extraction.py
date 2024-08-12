# This script iterates over all videos of yfcc database and extacts audio from their videos.
import os
import pandas as pd
from config import cfg
import subprocess
from zipfile import ZipFile
from tqdm import tqdm

output_dir = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/yfcc/raw_audio"
unzipped_videos_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/yfcc/unzipped_videos"
input_dir = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/yfcc/videos"

def extract_mp3(key):
    ZIPID = key[:3]
    subdir = key[3:6]
    mp4_path = "data/videos/mp4/{}/{}/{}.mp4".format(ZIPID,subdir,key)

    zip_path = os.path.join(input_dir,ZIPID+".zip")
    output_mp4_path = os.path.join(unzipped_videos_path,key)

    try:
        # loading the temp.zip and creating a zip object
        with ZipFile(zip_path, 'r') as zObject:
            # Extracting specific file in the zip
            # into a specific location.
            zObject.extract(
                mp4_path, path=output_mp4_path)
        zObject.close()
    except:
        status = "unzip_failed"
        return status
    mp4_to_convert = os.path.join(output_mp4_path,"data/videos/mp4/{}/{}/{}.mp4".format(ZIPID,subdir,key))
    mp3_filepath = os.path.join(output_dir,key+".mp3")
    try:
        cmd = 'ffmpeg -hide_banner -loglevel error -i {} {}'.format(mp4_to_convert, mp3_filepath)
        subprocess.run(cmd,shell=True)
    except:
        status = "mp3_failed"
        return status
    
    status = "success"

    return status


keys = list(pd.read_csv(os.path.join(cfg.yfcc_data_path,"yfcc100m_metadata.csv"))['key'])
status_csv = pd.DataFrame(columns=['key','status'])
done_keys = []
status_history = []
for key in tqdm(keys):
    status = extract_mp3(key)
    # import code;code.interact(local=dict(globals(), **locals()));
    done_keys.append(key)
    status_history.append(status)

status_csv['key'] = done_keys
status_csv['status'] = status_history

status_csv.to_csv(os.path.join(cfg.yfcc_data_path,"yfcc100m_status.csv"))