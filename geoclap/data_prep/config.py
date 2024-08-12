from yacs.config import CfgNode as CN
import os
import pandas as pd
cfg = CN()

cfg.DataRoot = '/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/'
cfg.aporee_meta = os.path.join(cfg.DataRoot,"aporee/final_metadata_with_captions.csv")
cfg.data_path = cfg.DataRoot

##############################################################################################################
cfg.freesound_geotagged_IDs_csv_path = os.path.join(os.path.join(cfg.DataRoot,"freesound/freesound_geotagged_IDs.csv"))
cfg.freesound_license_csv_path = os.path.join(os.path.join(cfg.DataRoot,"freesound/freesound_license.csv"))
cfg.freesound_api = ""

cfg.iNat_data_path  = os.path.join(cfg.DataRoot,"iNat")
cfg.iNat_audio_path = os.path.join(cfg.DataRoot,"iNat/raw_audio")
cfg.iNat_metadata_path = os.path.join(cfg.DataRoot,"iNat/iNaT_metadata.csv")

cfg.yfcc_metadata_path = os.path.join(cfg.DataRoot,"yfcc")
cfg.yfcc_data_path = os.path.join(cfg.DataRoot,"yfcc")

cfg.bingmap_api_key1 = "" #aayush
cfg.bingmap_api_key2 = "" #subash

cfg.bingmap_aporee = os.path.join(cfg.DataRoot,"aporee/images/bingmap")
cfg.bingmap_iNat = os.path.join(cfg.DataRoot,"iNat/images/bingmap")
cfg.bingmap_freesound = os.path.join(cfg.DataRoot,"freesound/images/bingmap")
cfg.bingmap_yfcc = os.path.join(cfg.DataRoot,"yfcc/images/bingmap")

cfg.merged_latlong_csv = os.path.join(cfg.DataRoot,"metafiles/GeoSound/source_balanced_geotagged_sounds.csv")
cfg.out_dir = cfg.DataRoot
