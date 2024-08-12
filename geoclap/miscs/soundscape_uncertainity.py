# Given a precomputed embeddings for satellite images for a region using the script `compute_sat_embeddings.py`,
# This script computes norm of std and stores them into a csv file containing id, lat, long, std_norm.
# These results will be useful in creating figures demonstrating changes in uncertainty as zoom-level increases.

import h5py as h5
import torch
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
import code
import pandas as pd
import os


def get_region_files(region_embeds_path):
      region_embeds = h5.File(region_embeds_path,"r")
      region_location = np.array(region_embeds.get('location'))
      region_lats = region_location[:,0]
      region_longs = region_location[:,1]

      sat_std_norm_1 = np.array(region_embeds.get('sat_std_norm_1'))
      sat_std_norm_3 = np.array(region_embeds.get('sat_std_norm_3'))
      sat_std_norm_5 = np.array(region_embeds.get('sat_std_norm_5'))

      region_keys = region_embeds.get('key')
      region_keys = [int(i.item()) for i in region_keys]
      
      return list(region_keys), list(region_lats), list(region_longs), list(sat_std_norm_1), list(sat_std_norm_3), list(sat_std_norm_5)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--region_embeds_path', type=str, default="/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/embeds/USA_embeds_with_GeoSound_pcmepp_bingmap.h5")
    
    args = parser.parse_args()
    k, lat, long, norm1, norm3, norm5 = get_region_files(region_embeds_path=args.region_embeds_path)
    df = pd.DataFrame(columns=['key','latitude','longitude','sat_std_norm_1','sat_std_norm_3','sat_std_norm_5'])
    df['key'] = k
    df['latitude'] = lat
    df['longitude'] = long
    df['sat_std_norm_1'] = norm1
    df['sat_std_norm_3'] = norm3
    df['sat_std_norm_5'] = norm5

    save_path = args.region_embeds_path.replace(".h5","_std_norm.csv")
    df.to_csv(save_path)
