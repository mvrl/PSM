# This script prepares webdataset shards for the soundingEarth database. For train/val/test splits.
import os
import pandas as pd
import torchaudio
import webdataset as wds
from argparse import ArgumentParser
import sys
import json
import numpy as np
from ..utilities import clap_data_processor

data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee"
output_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/SoundingEarth_for_mapping_fairsplit"
sources = ['aporee']
meta_columns = ['long_key', 'key','longitude', 'latitude', 'altitude', 'description', 'date_recorded',
       'date_uploaded', 'creator', 'licenseurl', 'title', 'mp3name', 'mp3mb', 'mp3seconds', 'mp3channels',
        'mp3bitrate', 'mp3samplerate', 'mp3file','caption']


def _get_processed_audio(audio,sr,model_type='CLAP'):
    if 'clap' in model_type.lower():
        out = dict(clap_data_processor.get_audio_clap(audio,sr,padding="repeatpad",truncation="fusion"))
    else:
        raise NotImplementedError("allowed audio_encoder types are :[baselineCLAP, nonprobCLAP, probCLAP]") 
    return out

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    else:
        return str(object)
    
class Dataset_soundscape():
    def __init__(self,                            
                 split="train",
                 overhead="googleEarth"):
        self.split = split
        if split == "train": 
            self.meta_df = pd.read_csv(os.path.join(data_path,"aporee_train_fairsplit_10km.csv"))
        if split == "val": 
            self.meta_df = pd.read_csv(os.path.join(data_path,"aporee_val_fairsplit_10km.csv"))
        if split == "test": 
            self.meta_df = pd.read_csv(os.path.join(data_path,"aporee_test_fairsplit_10km.csv"))
        self.overhead = overhead
        self.aporee_meta = pd.read_csv(os.path.join(data_path,"final_metadata_with_captions.csv"))
       
    def __len__(self):
        return len(self.meta_df) 
    def __getitem__(self,idx):
        sample = dict(self.meta_df.iloc[idx][meta_columns])
        mp3name = sample['mp3name']
        short_id = sample['key']
        long_id = sample['long_key']
        
        if self.overhead == 'googleEarth':
            image_path = os.path.join(data_path,'images','googleEarth',str(short_id)+'.jpg')
        elif self.overhead == 'sentinel':
            image_path = os.path.join(data_path,'images','sentinel_geoclap',str(short_id)+'.jpeg')
        elif self.overhead == 'bingmap':
            image_path = os.path.join(data_path,'images','bingmap_geoclap',str(long_id)+'.jpg')
        else:
            raise NotImplementedError("supported satellite image types are:[googleEarth, sentinel, bingmap]")
        
        with open(image_path,"rb") as stream:
            image = stream.read()
        
        audio_path =  os.path.join(data_path,'raw_audio',long_id,mp3name)
        audio, sr = torchaudio.load(audio_path)

        audio_mel1 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel2 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel3 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel4 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel5 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        
        sample['sampling_rate']=sr
        out_dict = {'meta_dict':json.dumps(sample, default=np_encoder),'image':image,
                    'audio_mel1':audio_mel1,
                    'audio_mel2':audio_mel2,
                    'audio_mel3':audio_mel3,
                    'audio_mel4':audio_mel4,
                    'audio_mel5':audio_mel5
                    }
        return out_dict

if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('--overhead', type=str, default="googleEarth")
    parser.add_argument('--split', type=str, default="test")
    args = parser.parse_args()
    dataset = Dataset_soundscape(split=args.split, overhead=args.overhead)

    if args.split == "train":
        tarfile_path = os.path.join(output_path,args.split+"_%d.tar")
        sink = wds.ShardWriter(tarfile_path, maxcount=10000, maxsize=50e9)
    else:
        tarfile_path = os.path.join(output_path,args.split+".tar")
        sink = wds.TarWriter(tarfile_path)
    
    for index, (out_dict) in enumerate(dataset):
        meta_dict = json.loads(out_dict['meta_dict'])
        if index%1000==0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        sink.write({
            "__key__":  meta_dict['long_key'],
            "meta.json": meta_dict,
            "image.jpg": out_dict['image'],
            "audio_mel1.pyd":out_dict['audio_mel1'],
            "audio_mel2.pyd":out_dict['audio_mel2'],
            "audio_mel3.pyd":out_dict['audio_mel3'],
            "audio_mel4.pyd":out_dict['audio_mel4'],
            "audio_mel5.pyd":out_dict['audio_mel5']
        })
    sink.close()