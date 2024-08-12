# Small-scale soundscape mapping:
# For a region grid, this script first downloads bingmap images of size 1500 x 1500 at zoom level 18 and creates soundscape maps for the region for given query

import torch
import torchaudio
from argparse import Namespace
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import numpy as np
import random
from ..engine import GeoCLAPModel
from ..utilities import clap_data_processor
from ..metrics import compute_csd_sims
import pandas as pd
from ..ckpt_paths import ckpt_cfg
from ..config import cfg
import math
from tqdm import tqdm
import urllib
from ..utilities.SatImage_transform import zoom_transform, sat_transform
from PIL import Image
import time
from torch.utils.data import Dataset, DataLoader

bingmap_api = cfg.bingmap_api_key1
BATCH_SIZE=1000
caption_sources = ["meta","qwen","pengi"]
audio_source_map = {'yfcc':0,'iNat':1, 'aporee':2,'freesound':3}
caption_source_map = {'meta':0,"qwen":1,"pengi":2}
image_gsd = {'sentinel':10, 'bingmap':0.6, 'googleEarth':0.2}

#download bingmap image
def download_images(args):
    region_file_path = args.region_file_path
    #download image
    zoom_level = 18
    # image settings
    im_size = "1500,1500"
    template_url = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%%s/%d?mapSize=%s&key=%s" % (zoom_level, im_size, bingmap_api)
    region_df = pd.read_csv(region_file_path)
    image_paths = []
    print("downloading {%d} images for the region",len(region_df))
    for i in tqdm(range(len(region_df))):
        # time.sleep(1)
        sample = region_df.iloc[i]
        tmp_loc = "%s,%s" % (sample.Y, sample.X)
        image_url = template_url % (tmp_loc)
        image_file = os.path.join(args.images_dir,str(sample.id)+"_"+ str(sample.Y) + "_"+ str(sample.X)+'.jpeg')
        sat_image_path = image_file
        image_paths.append(sat_image_path)
        try:
            urllib.request.urlretrieve(image_url, sat_image_path)
        except urllib.error.HTTPError as e:
            print("HTTP Error:", e.code)
            continue
        except Exception as e:
            print("Other Error:", e)
            continue
    print("all images for the region downloaded")
    return image_paths

def process_metadata(args, model_metadata_type):
    out_dict =   {
                    'audio_source':None,'caption_source':None,
                    'sat_zoom_level':torch.tensor(args.zoom_level).long(),
                    'time':None, 'month':None,'time_valid':None, 'month_valid':None
                  }
    #Prepare metadata:  
    if 'asource' in model_metadata_type:
        out_dict['audio_source'] = torch.tensor(audio_source_map[args.asource]).long()
    if 'tsource' in model_metadata_type:
        out_dict["caption_source"] =  torch.tensor(caption_source_map[args.tsource]).long()
    if 'time' in model_metadata_type:
        time_encode = torch.tensor([np.sin(2*np.pi*args.hour/23), np.cos(2*np.pi*args.hour/23)]).float()
        time_valid = torch.tensor(True).long()
        out_dict['time'] = time_encode
        out_dict['time_valid'] = time_valid
    if 'month' in model_metadata_type:
        month_encode = torch.tensor([np.sin(2*np.pi*args.month/12), np.cos(2*np.pi*args.month/12)]).float()
        month_valid = torch.tensor(True).long()
        out_dict['month'] = month_encode
        out_dict['month_valid'] = month_valid
    return out_dict

def process_sat_image(sat_image_path, sat_type="sentinel", zoom_level=1):
    zoom_tr = zoom_transform(zoom_level=zoom_level, sat_type=sat_type)
    sat_tr = sat_transform(is_train=False, input_size=224)

    sat_image = Image.open(sat_image_path)

    level_image = zoom_tr(sat_image)
    level_image = np.array(torch.permute(level_image,[1,2,0]))
    final_image = sat_tr(level_image)
    return final_image


class SatelliteDataset(Dataset):
    def __init__(self, image_paths, keys, lats, longs, sat_type, zoom_level):
        self.image_paths = image_paths
        self.keys = keys
        self.lats = lats
        self.longs = longs
        self.sat_type = sat_type
        self.zoom_level = zoom_level

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            key = self.keys[idx]
            lat = self.lats[idx]
            long = self.longs[idx]
            # Load and preprocess the image
            image = process_sat_image(sat_image_path=image_path, sat_type=self.sat_type, zoom_level=self.zoom_level)
            return image, torch.tensor(lat), torch.tensor(long), torch.tensor(key)
        except:
            return None

def collate_fn(batch):
    # Filter out samples that are None (i.e., failed to load)
    batch_image = [sample[0] for sample in batch if sample is not None]
    batch_lat = [sample[1] for sample in batch if sample is not None]
    batch_long = [sample[2] for sample in batch if sample is not None]
    batch_key = [sample[3] for sample in batch if sample is not None]
    return torch.stack(batch_image, dim=0), torch.stack(batch_lat, dim=0), torch.stack(batch_long, dim=0),torch.stack(batch_key, dim=0)

def move_to_device(size,item, device):
    if item != None:
        item = item.unsqueeze(0)
        if len(item.shape) == 1:
            item = item.expand(size).to(device)
        else:
            item = item.expand(size,item.shape[1]).to(device)
    return item

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class soundscape_mapping_demo(object):
    def __init__(self,args,ckpt_path,device,
                 text_queries=None, audio_queries=None,image_paths=None):
        super().__init__()
        assert (text_queries != None) or (audio_queries != None)
        self.args = args
        self.ckpt_path = ckpt_path
        self.region_file_path = args.region_file_path
        self.device = device
        self.sat_embeddings = {}
        self.image_paths = image_paths
        self.region_keys, self.region_lats, self.region_longs  = self.get_region_files()
        self.region_size = len(self.region_keys)
        self.query_type = args.query_type
        self.eval_metadata_type = args.metadata_type
        self.zoom_level = args.zoom_level
        self.text_queries, self.audio_queries = text_queries, audio_queries
        self.model, self.hparams = self.get_geoclap()
        
    def get_region_files(self):
        success_ids = [int(float(p.split("/")[-1].split("_")[0])) for p in self.image_paths]
        success_lats = [float(p.split("/")[-1].split("_")[1]) for p in self.image_paths]
        success_longs = [float(p.split("/")[-1].split("_")[2].replace(".jpeg",'')) for p in self.image_paths]
        return success_ids, success_lats, success_longs
    
    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        hparams["mode"] = "evaluate"
        hparams['meta_droprate'] = 0.0
        model = GeoCLAPModel(Namespace(**hparams)).to(self.device)
        model.load_state_dict(pretrained_weights,strict=False)
        geoclap = model.eval()
        #set all requires grad to false
        for params in geoclap.parameters():
            params.requires_grad=False
        model = geoclap 
        return model, hparams 
    
    def get_text_embeddings(self):
        # import code; code.interact(local=dict(globals(), **locals()))
        text_sample = clap_data_processor.get_text_clap(text=self.text_queries)
        if len(text_sample['attention_mask'].shape) == 1:
            text_sample['attention_mask'] = text_sample['attention_mask'].unsqueeze(0)
            text_sample['input_ids'] = text_sample['input_ids'].unsqueeze(0)
        text_sample['attention_mask'] = text_sample['attention_mask'].to(self.device)
        text_sample['input_ids'] = text_sample['input_ids'].to(self.device)  
        text_embeds = self.model.text_encoder(text_sample)
        return text_embeds['mean'], text_embeds['std']

    def get_audio_embeddings(self):
        audio_feats_mean = []
        audio_feats_std = []
        for a in self.audio_queries:
            audio, sr = torchaudio.load(a)
            audio_sample = clap_data_processor.get_audio_clap(track=audio, sr=sr)
            audio_sample['input_features'] = audio_sample['input_features'].to(self.device)
            audio_sample['is_longer'] = audio_sample['is_longer'].to(self.device)
            audio_embed = self.model.audio_encoder(audio_sample)
            audio_feats_mean.append(audio_embed['mean'])
            audio_feats_std.append(audio_embed['std'])
        audio_feats_mean = torch.cat(audio_feats_mean,axis=0)
        audio_feats_std = torch.cat(audio_feats_std,axis=0)
        return audio_feats_mean, audio_feats_std 

    def get_meta_query(self,curr_size,meta_query):
        #['audio_source', 'caption_source', 'latlong', 'month', 'month_valid', 'sat', 'sat_zoom_level', 'time', 'time_valid']
        out_dict =    {
                      'sat_zoom_level':move_to_device(curr_size,meta_query['sat_zoom_level'],self.device),
                      'audio_source':move_to_device(curr_size,meta_query['audio_source'],self.device),
                      'caption_source':move_to_device(curr_size,meta_query['caption_source'],self.device),
                      'month':move_to_device(curr_size,meta_query['month'],self.device),'month_valid':move_to_device(curr_size,meta_query['month_valid'],self.device),
                      'time':move_to_device(curr_size,meta_query['time'],self.device),'time_valid':move_to_device(curr_size,meta_query['time_valid'],self.device),
                      }
        return out_dict
    
      
    def get_sat_embeddings(self):
        processed_metadata = process_metadata(self.args,model_metadata_type=self.hparams['metadata_type'])
        dataset = SatelliteDataset(image_paths=self.image_paths,lats=self.region_lats,longs=self.region_longs,keys= self.region_keys,sat_type=self.args.sat_type, zoom_level=self.args.zoom_level)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,collate_fn=collate_fn)
        sat_embeddings_mean = []
        sat_embeddings_std = []
        sat_lats = []
        sat_longs = []
        sat_keys = []
        for images, lats, longs, keys in tqdm(data_loader):
            lats = [l.item() for l in lats]
            longs = [l.item() for l in longs]
            keys = [k.item() for k in keys]
            sat_lats = sat_lats + lats
            sat_longs = sat_longs + longs
            sat_keys = sat_keys + keys

            current_sat_images = images.to(self.device)
            curr_size = len(images)
            zoom_levels = [self.zoom_level]*curr_size
            input_res = torch.tensor([1.0*z*image_gsd[self.args.sat_type] for z in zoom_levels])
            sat_feats = self.model.sat_encoder.encoder.backbone(current_sat_images,input_res=input_res)
            current_sat_embeds = self.model.sat_encoder.encoder.projector(sat_feats)
            
            meta_batch = self.get_meta_query(curr_size,processed_metadata)

            meta_batch['latlong'] = torch.tensor([np.sin(np.pi*np.array(lats)/90), np.cos(np.pi*np.array(lats)/90), np.sin(np.pi*np.array(longs)/180), np.cos(np.pi*np.array(longs)/180)]).float().T.to(self.device)
            if self.eval_metadata_type != "none":
                sat_embeddings_fused = self.model.sat_encoder.encoder.meta_fuser(sat_embeddings=current_sat_embeds,audio_source=meta_batch['audio_source'], caption_source= meta_batch['caption_source'],latlong=meta_batch['latlong'],
                                                                              month=meta_batch["month"], time=meta_batch["time"],time_valid=meta_batch["time_valid"], month_valid=meta_batch["month_valid"], eval_meta=self.eval_metadata_type)
            else:
                sat_embeddings_fused = current_sat_embeds
            sat_embeddings_mean.append(self.model.sat_encoder.encoder.mean_head(sat_embeddings_fused))
            sat_embeddings_std.append(self.model.sat_encoder.encoder.std_head(sat_embeddings_fused))
        
        sat_embeddings_mean = torch.cat(sat_embeddings_mean,dim=0)
        sat_embeddings_std = torch.cat(sat_embeddings_std,dim=0)
        
        return sat_embeddings_mean, sat_embeddings_std, sat_lats, sat_longs, sat_keys

    
    def get_similarity_score(self):
        dists = {'for_text_query':None, 'for_audio_query':None, 'key':None,'latitude': None, 'longitude': None, 
                 'sat_embed_std_norm':None, 'audio_query_embed_std_norm':None, 'text_query_embed_std_norm':None}
        
        sat_embeds_mean, sat_embeds_std, sat_lats, sat_longs, sat_keys = self.get_sat_embeddings()
        
        dists['key'] = sat_keys
        dists['latitude'] = sat_lats
        dists['longitude'] = sat_longs

        dists['sat_embed_std_norm'] = list(torch.sqrt(torch.exp(torch.tensor(sat_embeds_std))).sum(dim=-1).detach().cpu().numpy())

        if 'audio' in self.query_type:
            audio_embeds_mean , audio_embeds_std  = self.get_audio_embeddings()
            dists['audio_query_embed_std_norm'] = list(torch.sqrt(torch.exp(torch.tensor(audio_embeds_std))).sum(dim=-1).detach().cpu().numpy())
            distance_matrix_audio = compute_csd_sims(l2normalize(sat_embeds_mean), l2normalize(audio_embeds_mean), sat_embeds_std, audio_embeds_std)
            dists['for_audio_query'] = np.array(distance_matrix_audio)

        if 'text' in self.query_type:
            text_embeds_mean , text_embeds_std  = self.get_text_embeddings()
            dists['text_query_embed_std_norm'] = list(torch.sqrt(torch.exp(torch.tensor(text_embeds_std))).sum(dim=-1).detach().cpu().numpy())
            distance_matrix_text = compute_csd_sims(l2normalize(sat_embeds_mean), l2normalize(text_embeds_mean), sat_embeds_std, text_embeds_std)
            dists['for_text_query'] = np.array(distance_matrix_text)
        return dists


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('--region_file_path', type=str, default="/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/maps/dummy/grid_points_dummy.csv")
    parser.add_argument('--images_downloaded', type=str, default="true",choices=["true","false"])
    parser.add_argument('--images_dir', type=str, default="/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/maps/dummy")
    parser.add_argument('--output_name', type=str, default="dummy_soundscape")
    parser.add_argument('--sat_type', type=str, default='bingmap')
    parser.add_argument('--month', type=int, default=5)
    parser.add_argument('--hour', type=int, default=14)
    parser.add_argument('--zoom_level', type=int, default=5)
    parser.add_argument('--asource', type=str, default="yfcc",choices=["yfcc","iNat","aporee","freesound"])
    parser.add_argument('--tsource', type=str, default="meta",choices=["meta","qwen","pengi"])
    parser.add_argument('--metadata_type', type=str, default='month_time_asource',help="'month', 'time', 'month_time','month_time_asource', 'month_time_asource_tsource', 'none'")
    parser.add_argument('--query_type', type=str, default='audio_text')
    parser.add_argument('--text_query_file', type=str, default='/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/maps/textual_query.txt')
    parser.add_argument('--audio_query_file', type=str, default='/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/maps/audio_query.txt')
    parser.add_argument('--expr', type=str, default='GeoSound_pcmepp_metadata_bingmap') #Options: #["GeoSound_infonce_bingmap","GeoSound_infonce_metadata_bingmap","GeoSound_pcmepp_bingmap","GeoSound_pcmepp_metadata_bingmap",
                                                                  #"GeoSound_infonce_sentinel","GeoSound_infonce_metadata_sentinel","GeoSound_pcmepp_sentinel","GeoSound_pcmepp_metadata_sentinel"]
           
    
    args = parser.parse_args()
    
    ckpt_path = ckpt_cfg[args.expr]
    if 'audio' in args.query_type:
        audio_query_file = args.audio_query_file
    else:
        audio_query_file = None
    if 'text' in args.query_type:
        text_query_file = args.text_query_file
    else:
        text_query_file = None

    text_queries = []
    audio_queries = []
    
    if text_query_file != None: 
        with open(text_query_file, 'r') as tfile:
            text_queries = tfile.read().strip().split("\n")
            

    if audio_query_file != None:
        with open(audio_query_file, 'r') as afile:
            audio_queries = afile.read().strip().split("\n")
            
    #download the images for the region
    if args.images_downloaded == "true":
        image_paths = [os.path.join(args.images_dir,p) for p in os.listdir(args.images_dir) if ".jpeg" in p]
    else:
        image_paths = download_images(args)
    demo_class = soundscape_mapping_demo(args= args,
                                         ckpt_path=ckpt_path,device=device,
                                         text_queries=text_queries, audio_queries=audio_queries,
                                         image_paths=image_paths)
    
    similarity_scores = demo_class.get_similarity_score()
    
    if 'audio' in args.query_type:
        df_audio_std_norm = pd.DataFrame(columns=['audio_query','audio_query_embed_std_norm'])
        df_audio_std_norm['audio_query'] = audio_queries
        df_audio_std_norm['audio_query_embed_std_norm'] = similarity_scores['audio_query_embed_std_norm']
        df_audio_std_norm.to_csv(os.path.join(args.images_dir, args.output_name+"_audioquery_std_norm.csv"))

        df_audio = pd.DataFrame(columns=['key','latitude','longitude','sat_embed_std_norm'])
        df_audio['key'] = list(similarity_scores['key'])
        df_audio['latitude'] = list(similarity_scores['latitude'])
        df_audio['longitude'] = list(similarity_scores['longitude'])
        df_audio['sat_embed_std_norm'] = list(similarity_scores['sat_embed_std_norm'])
        df_audio_dist = pd.DataFrame.from_records(similarity_scores['for_audio_query'])
        df_audio_dist.columns = [a.split("/")[-1] for a in audio_queries]
        df_audio = pd.concat([df_audio, df_audio_dist],axis=1)
        df_audio.to_csv(os.path.join(args.images_dir, args.output_name+"_audioquery.csv")) 
        print("Similarity for audio query saved to:",os.path.join(args.images_dir, args.output_name+"_audioquery.csv"))
    
    if 'text' in args.query_type:
        df_text_std_norm = pd.DataFrame(columns=['text_query','text_query_embed_std_norm'])
        df_text_std_norm['text_query'] = text_queries
        df_text_std_norm['text_query_embed_std_norm'] = similarity_scores['text_query_embed_std_norm']
        df_text_std_norm.to_csv(os.path.join(args.images_dir, args.output_name+"_textquery_std_norm.csv"))

        df_text = pd.DataFrame(columns=['key','latitude','longitude','sat_embed_std_norm'])
        df_text['key'] = list(similarity_scores['key'])
        df_text['latitude'] = list(similarity_scores['latitude'])
        df_text['longitude'] = list(similarity_scores['longitude'])
        df_text['sat_embed_std_norm'] = list(similarity_scores['sat_embed_std_norm'])

        df_text_dist = pd.DataFrame.from_records(similarity_scores['for_text_query'])
        df_text_dist.columns = text_queries
        df_text = pd.concat([df_text, df_text_dist],axis=1)
        df_text.to_csv(os.path.join(args.images_dir, args.output_name+"_textquery.csv"))  
        print("Similarity for text query saved to:",os.path.join(args.images_dir, args.output_name+"_textquery.csv"))
    # import code; code.interact(local=dict(globals(), **locals()))