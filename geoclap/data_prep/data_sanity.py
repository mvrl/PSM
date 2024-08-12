# A simple script to identify possibly corrupt files in the database.
import os
import pandas as pd
import torchaudio
from torchvision.io import read_image
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import torch
from config import cfg
from torch.utils.data import Dataset

data_path = cfg.data_path
def check_file(audio_path,source,key):
    try:
        wav, sr = torchaudio.load(audio_path)
        sentinel_img = read_image(os.path.join(cfg.data_path,source,'images',"sentinel",str(key)+'.jpeg'))
        bingmap_img = read_image(os.path.join(cfg.data_path,source,'images',"bingmap",str(key)+'.jpeg'))
        landcover_img = read_image(os.path.join(cfg.data_path,source,'images',"land_cover",str(key)+'.jpeg'))
        status = True
    except:
        print("failed for:",audio_path)
        status = False
        sr = 0
    
    return status, sr
         
class Dataset_sanity(Dataset):
    def __init__(self,                            
                 ):    
        self.overall_meta = pd.read_csv(os.path.join(data_path,"merged_metadata_final.csv"))
        self.aporee_meta = pd.read_csv(os.path.join(data_path,"aporee","final_metadata_with_captions.csv"))
       
    def __len__(self):
        return len(self.overall_meta)
    def __getitem__(self,idx):
        sample = self.overall_meta.iloc[idx]
        source = sample.source
        key = sample.key
        
        if source == 'iNat':
            sound_format = sample.sound_format
        else:
            sound_format =  'mp3'
        if source == 'aporee':   
            soundname = self.aporee_meta[self.aporee_meta['long_key']==key].mp3name.item()
            audio_path = os.path.join(cfg.data_path,source,'raw_audio',str(key),soundname)
        else:
            if isinstance(key, str):
                soundname = key+"."+sound_format
            else:
                soundname = str(key)+"."+sound_format
            
            audio_path = os.path.join(cfg.data_path,source,'raw_audio',soundname)
            
        status, sr = check_file(audio_path,source,key)
        out_dict = {'source':source,'key':str(key),'soundname':soundname,'original_sampling_rate':sr,'status':status}
        return out_dict


source = []
key = []
soundname = []
original_sampling_rate = []
status = []

dataloader = torch.utils.data.DataLoader(Dataset_sanity(),num_workers=32, batch_size=256, shuffle=True, drop_last=False,pin_memory=True)
for batch in tqdm(dataloader):
    # import code;code.interact(local=dict(globals(), **locals()));
    source = source + batch['source']
    key =  key + batch['key']
    soundname = soundname + batch['soundname']
    original_sampling_rate = original_sampling_rate + batch['original_sampling_rate'].tolist()
    status = status + batch['status'].tolist()

df = pd.DataFrame(columns=['source','key','soundname','original_sampling_rate','status'])

df['source'] = source
df['key'] = key
df['soundname'] = soundname
df['original_sampling_rate'] = original_sampling_rate
df['status'] = status

df.to_csv(os.path.join(data_path,"dataset_sanity.csv"))