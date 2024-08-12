# This script prepares webdataset shards for our database. For train/val/test splits we create seperate set of datasets for bingmap and sentinel imagery. 
import os
import pandas as pd
import torchaudio
import webdataset as wds
from argparse import ArgumentParser
import sys
import json
import numpy as np
from ..utilities import clap_data_processor

data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/"
output_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_compressed/data_compressed/GeoSound_for_mapping/"
sources = ['iNat', 'yfcc', 'aporee', 'freesound']
meta_columns = ['sample_id','date', 'latitude','longitude', 'description', 'tags', 
                'title', 'scientific_name', 'common_name', 'sound_format', 'text',
                'address', 'original_sampling_rate', 'bin_id']


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
                 overhead="sentinel"):
        self.split = split
       
        if split == "train": 
            self.meta_df = pd.read_csv(os.path.join(data_path,"/metafiles/GeoSound/train_metadata.csv"))
        if split == "val": 
            self.meta_df = pd.read_csv(os.path.join(data_path,"/metafiles/GeoSound/val_metadata.csv"))
        if split == "test": 
            self.meta_df = pd.read_csv(os.path.join(data_path,"/metafiles/GeoSound/test_metadata.csv"))
        self.overhead = overhead
        self.aporee_meta = pd.read_csv(os.path.join(data_path,"/aporee/final_metadata_with_captions.csv"))
       
    def __len__(self):
        return len(self.meta_df) 
    def __getitem__(self,idx):
        sample = dict(self.meta_df.iloc[idx][meta_columns])
        sample_id = sample['sample_id']
        source = sample_id.split("-")[0]
        key = sample_id.split("-")[1]
        sound_format =  'mp3'
        if source == 'aporee':   
            soundname = self.aporee_meta[self.aporee_meta['long_key']==key].mp3name.item()
            audio_path = os.path.join(data_path,source,'raw_audio',str(key),soundname)
        else:
            if isinstance(key, str):
                soundname = key+"."+sound_format
            else:
                soundname = str(key)+"."+sound_format
            
            audio_path = os.path.join(data_path,source,'raw_audio',soundname)
        if self.overhead == "sentinel":
            image_path = os.path.join(data_path,source,'images',"sentinel",str(key)+'.jpeg')
        if self.overhead == "bingmap":
            image_path = os.path.join(data_path,source,'images',"bingmap",str(key)+'.jpeg')

        with open(image_path,"rb") as stream:
            image = stream.read()
        
        audio, sr = torchaudio.load(audio_path)
        
        audio_mel1 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel2 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel3 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel4 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        audio_mel5 = _get_processed_audio(audio,sr)['input_features'][0,:,:,:]
        
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
    parser.add_argument('--overhead', type=str, default="sentinel")
    parser.add_argument('--split', type=str, default="test")
    args = parser.parse_args()
    dataset = Dataset_soundscape(split=args.split, overhead=args.overhead)
    if args.split == "train":
        tarfile_path = os.path.join(output_path,"sat2sound_webdataset_final","with_"+args.overhead,args.split+ "_"+args.overhead+"_%db.tar")
        sink = wds.ShardWriter(tarfile_path, maxcount=50000, maxsize=50e9)
    else:
        tarfile_path = os.path.join(output_path,"sat2sound_webdataset_final","with_"+args.overhead,args.split+ "_"+args.overhead+".tar")
        sink = wds.TarWriter(tarfile_path)
    
    for index, (out_dict) in enumerate(dataset):
        meta_dict = json.loads(out_dict['meta_dict'])
        audio_format = meta_dict['sound_format']
        if index%1000==0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        sink.write({
            "__key__":  meta_dict['sample_id'],
            "meta.json": meta_dict,
            "image.jpg": out_dict['image'],
            "audio_mel1.pyd":out_dict['audio_mel1'],
            "audio_mel2.pyd":out_dict['audio_mel2'],
            "audio_mel3.pyd":out_dict['audio_mel3'],
            "audio_mel4.pyd":out_dict['audio_mel4'],
            "audio_mel5.pyd":out_dict['audio_mel5']
        })
    sink.close()
