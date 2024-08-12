import os
import errno
import urllib.request
from multiprocessing import Pool
import time 
import signal
import code
import argparse
from config import cfg
import pandas as pd
from tqdm import tqdm

LIMIT = 50000 #Download limit per job

def timeout_handler(signum, frame):
    raise TimeoutError("Program timed out")
    
def ensure_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise

def download(url, out_file):
  if not os.path.isfile(out_file):
    # time.sleep(1)
    ensure_dir(out_file)
    try:
      urllib.request.urlretrieve(url, out_file)
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code)
        raise
    except Exception as e:
        print("Other Error:", e)
        raise
        
if __name__ == '__main__':
  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(50000)

  parser = argparse.ArgumentParser(description='Download satellite imagery from bingmaps')
  parser.add_argument('--api_key', type=str,default="",
                    help='Bingmap api key')
  parser.add_argument('--merged_latlong_csv', type=str,default="/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/source_balanced_geotagged_sounds.csv",
                    help='csv file containing lat, long for the images to download')
  parser.add_argument('--out_dir', type=str,default=cfg.out_dir,
                    help='csv file containing lat, long for the images to download')
  parser.add_argument('--split_id', type=str,
                    help='Split ID for blocks of $LIMIT sound IDs')
  
  args = parser.parse_args()
  
  out_dir = args.out_dir
  meta_df = pd.read_csv(args.merged_latlong_csv,low_memory=False)
  # Split sound IDs into blocks of $LIMIT
  split_id = int(args.split_id)
  start_index = (split_id - 1) * LIMIT
  end_index = split_id * LIMIT
  sub_df = meta_df.iloc[start_index:end_index]
  zoom_level = 18
  # settings
  im_size = "1500,1500"
  api_key = args.api_key
  jobs = []

  template_url = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%%s/%d?mapSize=%s&key=%s" % (zoom_level, im_size, api_key)
  
  for i in tqdm(range(len(sub_df))):
     row = sub_df.iloc[i]
     lat = row.latitude
     lon = row.longitude
     key = row.key
     source = row.source
     tmp_loc = "%s,%s" % (lat, lon)
     image_url = template_url % (tmp_loc)
    # code.interact(local=dict(locals(), **globals()))
     out_file = f'{out_dir}/{source}/{"images/bingmap"}/{key}.jpeg'
     download(url=image_url,out_file=out_file)
