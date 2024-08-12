from yacs.config import CfgNode as CN
import os
import pandas as pd
cfg = CN()

cfg.DataRoot = '/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/'
cfg.pretrained_models_path = '/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/'
cfg.satmae_pretrained_ckpt = os.path.join(cfg.pretrained_models_path,'SATMAE','pretrain-vit-base-e199.pth')
cfg.aporee_meta = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/aporee/final_metadata_with_captions.csv"
cfg.data_path = cfg.DataRoot
cfg.log_dir = "/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs"
##############################################################################################################
################################### WEBDATASET PATHS ###########################################################
cfg.GeoSound_webdataset_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/GeoSound_for_mapping"
cfg.SoundingEarth_webdataset_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/SoundingEarth_for_mapping_fairsplit"

cfg.llava_caption_for_sentinel = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/GeoSound_for_mapping/llava_caption_for_sentinel.json"
cfg.llava_caption_for_bingmap = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/GeoSound_for_mapping/llava_caption_for_bingmap.json"
cfg.llava_caption_for_SoundingEarth = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/SoundingEarth_for_mapping_fairsplit/SoundingEarth_llava_caption_for_googleEarth.json"