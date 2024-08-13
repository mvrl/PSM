from yacs.config import CfgNode as CN
import os
import pandas as pd
cfg = CN()

cfg.DataRoot = 'path_to_data'
cfg.pretrained_models_path = 'path_to_SATMAE_checkpoint_dir'
cfg.satmae_pretrained_ckpt = os.path.join(cfg.pretrained_models_path,'pretrain-vit-base-e199.pth')
cfg.data_path = cfg.DataRoot
cfg.log_dir = "path_to/logs"
##############################################################################################################
################################### WEBDATASET PATHS ###########################################################
cfg.GeoSound_webdataset_path = "path_to/GeoSound_for_mapping"
cfg.SoundingEarth_webdataset_path = "path_to/SoundingEarth_for_mapping_fairsplit"

cfg.llava_caption_for_sentinel = "path_to/llava_caption_for_sentinel.json"
cfg.llava_caption_for_bingmap = "path_to/llava_caption_for_bingmap.json"
cfg.llava_caption_for_SoundingEarth = "path_to/SoundingEarth_llava_caption_for_googleEarth.json"