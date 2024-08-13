#This dataloader works under the assumption that separate datasets are created for sentinel vs bingmap imagery.
import numpy as np
import webdataset as wds
import torch
import code
from argparse import Namespace
import random
import time
from .utilities.SatImage_transform import zoom_transform, sat_transform
from .utilities import clap_data_processor
from .utilities.date_cleanup import get_clean_date
from .config import cfg
import os
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd

caption_sources = ["meta","qwen","pengi"]
audio_source_map = {'yfcc':0,'iNat':1, 'aporee':2,'freesound':3}
caption_source_map = {'meta':0,"qwen":1,"pengi":2}

clap_score_df = pd.read_csv(os.path.join(cfg.data_path,"clap_score_geosound.csv"))
pengi_caption = pd.read_json(os.path.join(cfg.data_path,"geosound_audio_caption_pengi.json"),lines=True)
qwen_caption = pd.read_json(os.path.join(cfg.data_path,"geosound_audio_caption_qwen.json"),lines=True)

## One way to implement modality dropout:
def dropout(droprate=0.5):
    percent = droprate *100
    return random.randrange(0, 100) < percent


def _get_processed_text(text, model_type='CLAP'):
    if 'clap' in model_type.lower():
        out = dict(clap_data_processor.get_text_clap([text]))
    else:
        raise NotImplementedError("only allowed text_encoder types are from :[CLAP]")
    return out

# def _get_processed_audio(audio,sr,model_type='CLAP'):
#     if 'clap' in model_type.lower():
#         out = dict(clap_data_processor.get_audio_clap(audio,sr,padding="repeatpad",truncation="fusion"))
#     else:
#         raise NotImplementedError("only allowed audio_encoder types are from :[CLAP]") 
#     return out

class Dataset_soundscape(object):
    def __init__(self,
                 args,
                 is_train = True,                            # train/val/test
                 test_zoom_level = None,
                 test_mel_index = None,
                 ):
       
        self.args = args
        self.is_train = is_train
        self.dataset_type = args.dataset_type
        self.modality_type = args.modality_type
        self.sat_transform = sat_transform(is_train=self.is_train, input_size=args.sat_input_size)
        self.metadata_type = args.metadata_type
        self.sat_type = args.sat_type
        self.test_zoom_level = test_zoom_level
        self.test_mel_index = test_mel_index #5 random mel features were saved so index to select index during evaluation

    def get_ds(self,mode):
        print(f'\nInitializing {mode} dataset')
        if mode=='train':
            print("Initializing train webdataset for train epoch length:",self.args.train_epoch_length)
            self.dataset = wds.WebDataset(self.args.train_path, resampled=True)
            self.dataset = self.dataset.shuffle(1000).decode("pil", handler=wds.warn_and_continue).to_tuple("meta.json", "image.jpg", "audio_mel1.pyd","audio_mel2.pyd","audio_mel3.pyd","audio_mel4.pyd","audio_mel5.pyd","__key__").map(self.do_transforms, handler=wds.warn_and_continue).batched(self.args.train_batch_size).with_epoch(self.args.train_epoch_length)
        
        elif mode=='val':
            self.dataset = wds.WebDataset(self.args.vali_path)
            self.dataset = self.dataset.decode("pil", handler=wds.warn_and_continue).to_tuple("meta.json", "image.jpg", "audio_mel1.pyd","audio_mel2.pyd","audio_mel3.pyd","audio_mel4.pyd","audio_mel5.pyd","__key__").map(self.do_transforms, handler=wds.warn_and_continue).batched(self.args.val_batch_size)
        
        elif mode=='test':
            self.dataset = wds.WebDataset(self.args.test_path)
            self.dataset = self.dataset.decode("pil", handler=wds.warn_and_continue).to_tuple("meta.json", "image.jpg", "audio_mel1.pyd","audio_mel2.pyd","audio_mel3.pyd","audio_mel4.pyd","audio_mel5.pyd","__key__").map(self.do_transforms, handler=wds.warn_and_continue).batched(self.args.test_batch_size)
        return self.dataset
        
    
    def do_transforms(self, sample):
        out_dict =   {'key':None,
                      'audio_source':None,'caption_source':None,
                      'audio_input_features':None, 'audio_is_longer':None, 
                      'text_input_ids':None,'text_attention_mask':None,
                      'sat_zoom_level':None,'sat':None,'latlong':None,
                      'time':None, 'month':None,'time_valid':None, 'month_valid':None}
        
       
        meta, image, mel1,mel2,mel3,mel4,mel5, key = sample
        if self.dataset_type == "SoundingEarth":
            key = "aporee-"+key
            self.sat_type = "googleEarth" # Experiment with SoundingEarth contains only googleEarth imagery.
        ################################################################################
        #prepare sat_image:
        if self.test_zoom_level != None:
            zoom_level = self.test_zoom_level
        else:
            if self.sat_type != "googleEarth":
                zoom_level = random.choice([1,3,5])
            else:
                zoom_level = 1

        sat_image = image
        out_dict['sat_zoom_level'] = torch.tensor(zoom_level).long()
        zoom_tr = zoom_transform(zoom_level=zoom_level, sat_type=self.sat_type)
        level_image = zoom_tr(sat_image)
        level_image = np.array(torch.permute(level_image,[1,2,0]))
        final_image = self.sat_transform(level_image)
        out_dict['sat']= final_image
        ################################################################################
        #prepare audio feature:
        mels = [mel1,mel2,mel3,mel4,mel5]
        if self.test_mel_index == None:
            mel_index = random.choice([0,1,2,3,4])
        else:
            mel_index = self.test_mel_index
        audio_mel = mels[mel_index]
        out_dict['audio_input_features'] = audio_mel
        out_dict['audio_is_longer'] = torch.tensor(True).long()
        ################################################################################
        #prepare text feature:
        caption_source = clap_score_df[clap_score_df["sample_id"]==key]["best_caption"].item() #find which audio caption is best based on CLAP score.
        if self.args.caption_strategy == "original":
            if caption_source == "pengi":
                caption = pengi_caption[pengi_caption["sample_id"]==key]["pengi_caption"].item()
            if caption_source == "qwen":
                caption = qwen_caption[qwen_caption["sample_id"]==key]["qwen_caption"].item()
            else:
                if self.dataset_type == "GeoSound":
                    caption = meta["text"]
                else:
                    caption = meta['caption'].split("The location of the sound is")[0] + "."
        else:
            if self.args.caption_strategy == "pengi":
                caption_source = "pengi"
                caption = pengi_caption[pengi_caption["sample_id"]==key]["pengi_caption"].item()
            elif self.args.caption_strategy == "qwen":
                caption_source = "qwen"
                caption = qwen_caption[qwen_caption["sample_id"]==key]["qwen_caption"].item()
            elif self.args.caption_strategy == "meta":
                caption_source = "meta"
                if self.dataset_type == "GeoSound":
                    caption = meta["text"]
                else:
                    caption = meta['caption'].split("The location of the sound is")[0] + "."
            else:
                raise NotImplementedError("Caption strategy not implemented")
            
        if 'text' in self.modality_type:
            out_dict_text = _get_processed_text(text=caption)
            out_dict['text_attention_mask'] = out_dict_text['attention_mask']
            out_dict['text_input_ids'] = out_dict_text['input_ids']
        ################################################################################
        #Prepare metadata:
        out_dict['key'] = key
        long = meta['longitude']
        lat = meta['latitude']
        latlong_encode = torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*long/180), np.cos(np.pi*long/180)]).float()
        
        if self.dataset_type == "GeoSound":
            date = get_clean_date(meta['date'])
            source = key.split("-")[0]
        else:
            source = "aporee"
            date = get_clean_date(meta['date_recorded'])
        
        if source == "freesound": # No time information for freesound samples so..
            time_encode = torch.tensor([0., 0.]).float()
            time_valid = torch.tensor(False).long()
        else:
            if date is not None:
                time_encode = torch.tensor([np.sin(2*np.pi*date.hour/23), np.cos(2*np.pi*date.hour/23)]).float()
                time_valid = torch.tensor(True).long()
            else:
                time_encode = torch.tensor([0., 0.]).float()
                time_valid = torch.tensor(False).long()

        #month encoding
        if date is not None:
            month_encode = torch.tensor([np.sin(2*np.pi*date.month/12), np.cos(2*np.pi*date.month/12)]).float()
            month_valid = torch.tensor(True).long()
        else:
            month_encode = torch.tensor([0., 0.]).float()
            month_valid = torch.tensor(False).long()
        
        if 'asource' in self.metadata_type:
            out_dict['audio_source'] = torch.tensor(audio_source_map[source]).long()   
        if 'tsource' in self.metadata_type:
            out_dict["caption_source"] =  torch.tensor(caption_source_map[caption_source]).long()
        if 'latlong' in self.metadata_type:
            out_dict['latlong'] = latlong_encode
        if 'time' in self.metadata_type:
            out_dict['time'] = time_encode
            out_dict['time_valid'] = time_valid
        if 'month' in self.metadata_type:
            out_dict['month'] = month_encode
            out_dict['month_valid'] = month_valid
        
        return [out_dict['key'], out_dict['audio_source'], out_dict['caption_source'], out_dict['audio_input_features'], out_dict['audio_is_longer'], out_dict['text_input_ids'], out_dict['text_attention_mask'], out_dict['sat_zoom_level'], out_dict['sat'], out_dict['latlong'], out_dict['time'], out_dict['month'],out_dict['time_valid'], out_dict['month_valid']]

def get_shards(dataset_type="GeoSound",overhead_type="sentinel"):
    if dataset_type == "GeoSound":
        data_path = os.path.join(cfg.GeoSound_webdataset_path,"with_"+overhead_type)
    else:
        data_path = cfg.SoundingEarth_webdataset_path
    all_shards = [os.path.join(data_path,s) for s in os.listdir(data_path) if ".tar" in s]
    test_shard = [s for s in all_shards if 'test' in s]
    val_shard = [s for s in all_shards if 'val' in s]
    train_shards = [s for s in all_shards if 'train' in s]
    return train_shards, val_shard, test_shard

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('--dataset_type', type=str, default='GeoSound',choices=['GeoSound','SoundingEarth'])
    parser.add_argument('--modality_type', type=str, default='sat_audio_text')
    parser.add_argument('--sat_input_size', type=int, default= 224)
    parser.add_argument('--sat_type', type=str, default='sentinel', choices=['sentinel','bingmap','googleEarth'])
    parser.add_argument('--metadata_type', type=str, default='latlong_month_time_asource_tsource',choices=['latlong', 'month', 'latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource', 'none'])
    args = parser.parse_args()
    print(args)
    train_shards, val_shard, test_shard = get_shards(dataset_type=args.dataset_type,overhead_type=args.sat_type)
    args = {     'train_path': train_shards,
                 'train_batch_size': 14,
                 'val_batch_size' : 14,
                 'test_batch_size' : 14,
                 'train_epoch_length': 10,
                 'sat_input_size': 224,                     # Input size of satellite image
                 'modality_type':args.modality_type,        # data choices: [sat_audio, sat_audio_meta, sat_audio_text]
                 'sat_encoder_type':'probScaleMAE',         # Choice of satellite image model: [baselineScaleMAE, nonprobScaleMAE, probScaleMAE]
                 'audio_encoder_type':'probCLAP',           # Choice of text_audio model: [baselineCLAP, nonprobCLAP, probCLAP]
                 'metadata_type':args.metadata_type,        # What extra metadata to pass, currently only supports: ['latlong', 'month', 'latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource', 'none']
                 'sat_type':args.sat_type,                  # what type of satellite image to use: [sentinel, bingmap]
                 'dataset_type':args.dataset_type,
                 'caption_strategy':"original"
            }
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            args = Namespace(**args)
            dataset = Dataset_soundscape(args).get_ds('train')
            out_dict = {}
            tick = time.time()
            for i, sample in enumerate(dataset):
                audio_feat_dict = {}
                text_feat_dict = {}
                out_dict['key'], out_dict['audio_source'], out_dict['caption_source'], out_dict['audio_input_features'], out_dict['audio_is_longer'], out_dict['text_input_ids'], out_dict['text_attention_mask'], out_dict['sat_zoom_level'], out_dict['sat'], out_dict['latlong'], out_dict['time'], out_dict['month'],out_dict['time_valid'], out_dict['month_valid']= sample
                print(f'Sample no {i}')
                code.interact(local=dict(globals(), **locals()))
                if i == 3:
                    break
            tock = time.time()
            time_taken = tock - tick
    print(f'The total time taken is {time_taken}')
    # code.interact(local=dict(globals(), **locals()))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))