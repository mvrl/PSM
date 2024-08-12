# For a satellite image retreieve topk audio samples
from ..metrics import compute_csd_sims
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import numpy as np
import torch
from ..engine import GeoCLAPModel
import pandas as pd
import random
from ..utilities.SatImage_transform import zoom_transform, sat_transform
from argparse import Namespace
import h5py as h5
import urllib
from ..config import cfg
from PIL import Image
from ..ckpt_paths import ckpt_cfg

bingmap_api = cfg.bingmap_api_key2

data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/"
aporee_meta = pd.read_csv(os.path.join(data_path,"aporee","final_metadata_with_captions.csv"))
gallery_meta = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/test_metadata.csv")
caption_sources = ["meta","qwen","pengi"]
audio_source_map = {'yfcc':0,'iNat':1, 'aporee':2,'freesound':3}
caption_source_map = {'meta':0,"qwen":1,"pengi":2}

pengi_caption = pd.read_json(os.path.join(cfg.GeoSound_webdataset_path,"geosound_audio_caption_pengi.json"),lines=True)
qwen_caption = pd.read_json(os.path.join(cfg.GeoSound_webdataset_path,"geosound_audio_caption_qwen.json"),lines=True)

def get_audio_caption(sample_id,caption_source="meta"):
    if caption_source == "pengi":
        caption = pengi_caption[pengi_caption["sample_id"]==sample_id]["pengi_caption"].item()
    elif caption_source == "qwen":
        caption = qwen_caption[qwen_caption["sample_id"]==sample_id]["qwen_caption"].item()
    else:
        caption = gallery_meta[gallery_meta["sample_id"]==sample_id]["text"].item()

    source = sample_id.split("-")[0]
    key = sample_id.split("-")[1]
    sound_format =  'mp3'
    if source == 'aporee':   
        soundname = aporee_meta[aporee_meta['long_key']==key].mp3name.item()
        audio_path = os.path.join(data_path,source,'raw_audio',str(key),soundname)
    else:
        if isinstance(key, str):
            soundname = key+"."+sound_format
        else:
            soundname = str(key)+"."+sound_format
        
        audio_path = os.path.join(data_path,source,'raw_audio',soundname)
    return {'audio':audio_path,'caption':caption}

def get_gallery_embeds(gallery_embeds_path):
    gallery_embeds = h5.File(gallery_embeds_path,"r")
    gallery_keys = list(gallery_embeds.get('key'))
    gallery_keys = [str(gallery_keys[i]).split("'")[1] for i in range(len(gallery_keys))]

    sat_embeddings = torch.tensor(np.array(gallery_embeds.get('sat_feat')))

    audio_embeddings_mean = torch.tensor(np.array(gallery_embeds.get('audio_embeddings_mean')))
    audio_embeddings_std = torch.tensor(np.array(gallery_embeds.get('audio_embeddings_std')))

    text_embeddings_mean = torch.tensor(np.array(gallery_embeds.get('text_embeddings_mean')))
    text_embeddings_std = torch.tensor(np.array(gallery_embeds.get('text_embeddings_std')))

    return gallery_keys, sat_embeddings,{"mean":audio_embeddings_mean, "std":audio_embeddings_std}, {"mean":text_embeddings_mean, "std":text_embeddings_std}
    

#download bingmap image
def download(url, out_file):
    try:
        urllib.request.urlretrieve(url, out_file)
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code)
        raise
    except Exception as e:
        print("Other Error:", e)
        raise

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

def move_to_device(item, device):
    if item!= None:
        item = item.to(device).unsqueeze(0)
    return item


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


def process_sat_image(sat_image_path, sat_type="sentinel", zoom_level=1):
    zoom_tr = zoom_transform(zoom_level=zoom_level, sat_type=sat_type)
    sat_tr = sat_transform(is_train=False, input_size=224)

    sat_image = Image.open(sat_image_path)

    level_image = zoom_tr(sat_image)
    level_image = np.array(torch.permute(level_image,[1,2,0]))
    final_image = sat_tr(level_image)
    return final_image
    
def process_metadata(args):
    out_dict =   {
                    'audio_source':None,'caption_source':None,
                    'sat_zoom_level':torch.tensor(args.zoom_level).long(),'sat':None,'latlong':None,
                    'time':None, 'month':None,'time_valid':None, 'month_valid':None
                  }
    #Prepare metadata:
    long = args.long
    lat = args.lat
       
    if 'asource' in args.metadata_type:
        out_dict['audio_source'] = torch.tensor(audio_source_map[args.asource]).long()   
    if 'tsource' in args.metadata_type:
        out_dict["caption_source"] =  torch.tensor(caption_source_map[args.tsource]).long()
    if 'latlong' in args.metadata_type:
        latlong_encode = torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*long/180), np.cos(np.pi*long/180)]).float()
        out_dict['latlong'] = latlong_encode
    if 'time' in args.metadata_type:
        time_encode = torch.tensor([np.sin(2*np.pi*args.hour/23), np.cos(2*np.pi*args.hour/23)]).float()
        time_valid = torch.tensor(True).long()
        out_dict['time'] = time_encode
        out_dict['time_valid'] = time_valid
    if 'month' in args.metadata_type:
        month_encode = torch.tensor([np.sin(2*np.pi*args.month/12), np.cos(2*np.pi*args.month/12)]).float()
        month_valid = torch.tensor(True).long()
        out_dict['month'] = month_encode
        out_dict['month_valid'] = month_valid
    
    return out_dict

class audio_retrieval_engine(object):
    def __init__(self,args,ckpt_path,audio_gallery_embeds_path,metadata_type,device):
        super().__init__()
        self.args = args
        self.ckpt_path = ckpt_path
        self.device = device
        self.metadata_type = metadata_type
        self.model, self.hparams = self.get_geoclap()
        self.gallery_keys, self.sat_embeddings, self.audio_embeddings, self.text_embeddings = get_gallery_embeds(audio_gallery_embeds_path)

    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        hparams["mode"] = "evaluate"
        hparams['metadata_type'] = self.metadata_type
        hparams['meta_droprate'] = 0.0
        model = GeoCLAPModel(Namespace(**hparams)).to(self.device)
        model.load_state_dict(pretrained_weights,strict=False)
        geoclap = model.eval()
        #set all requires grad to false
        for params in geoclap.parameters():
            params.requires_grad=False
        model = geoclap 
        return model, hparams 
    
    def get_sat_query(self,sat_query):
        #['audio_source', 'caption_source', 'latlong', 'month', 'month_valid', 'sat', 'sat_zoom_level', 'time', 'time_valid']
        out_dict =    {
                      'sat_zoom_level':move_to_device(sat_query['sat_zoom_level'],self.device),
                      'sat':move_to_device(sat_query['sat'],self.device),'latlong':move_to_device(sat_query['latlong'],self.device),
                      'audio_source':move_to_device(sat_query['audio_source'],self.device),
                      'caption_source':move_to_device(sat_query['caption_source'],self.device),
                      'month':move_to_device(sat_query['month'],self.device),'month_valid':move_to_device(sat_query['month_valid'],self.device),
                      'time':move_to_device(sat_query['time'],self.device),'time_valid':move_to_device(sat_query['time_valid'],self.device),
                      }
        return out_dict
    
    def get_sat_embeds(self,sat_query):
        batch = self.get_sat_query(sat_query)
        sat_embeddings =  self.model.sat_encoder(images=batch['sat'], zoom_level=batch['sat_zoom_level'],sat_type=self.hparams['sat_type'],
                                                 audio_source=batch['audio_source'], caption_source=batch['caption_source'], 
                                                 latlong=batch['latlong'], time= batch['time'], month = batch['month'], 
                                                 time_valid=batch['time_valid'], month_valid=batch['month_valid'], eval_meta=self.metadata_type) 
        return {'mean':sat_embeddings['mean'], 'std':sat_embeddings['std']}
    
    def get_topk_audios(self,sat_query,k=5):
        sat_embeds = self.get_sat_embeds(sat_query=sat_query)  
        gallery_embeds = {'mean':self.audio_embeddings['mean'], 'std':self.audio_embeddings['std']}
       
        distance_matrix = compute_csd_sims(l2normalize(gallery_embeds['mean']), l2normalize(sat_embeds['mean']), gallery_embeds['std'],sat_embeds['std'])
        topk_scores = torch.topk(torch.tensor(distance_matrix),k,dim=0)
        top_keys = [self.gallery_keys[i.item()] for i in topk_scores.indices][:self.args.samples]
        top_retreived = [get_audio_caption(sample_id,caption_source=self.args.tsource) for sample_id in top_keys]
        return top_retreived


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--gallery_embeds_path', type=str, default="/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/retrieval/GeoSound_pcmepp_metadata_bingmap_zoom_level_5_gallery.h5")
    parser.add_argument('--sat_query_path', type=str, default="")
    
    parser.add_argument('--output_dir', type=str, default="/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/retrieval")
    parser.add_argument('--output_name', type=str, default="demo")
    parser.add_argument('--sat_type', type=str, default='bingmap',choices=["sentinel","bingmap"])
    parser.add_argument('--zoom_level', type=int, default=5)
    
    parser.add_argument('--lat',type=float, default=25.772375)
    parser.add_argument('--long',type=float, default=-80.130166)
    parser.add_argument('--month', type=int, default=5)
    parser.add_argument('--hour', type=int, default=10)
    parser.add_argument('--asource', type=str, default="yfcc",choices=["yfcc","iNat","aporee","freesound"])
    parser.add_argument('--tsource', type=str, default="pengi",choices=["meta","qwen","pengi"])
    parser.add_argument('--metadata_type', type=str, default='latlong_month_time_asource_tsource',help="'latlong', 'month', 'latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource', 'none'")
    parser.add_argument('--expr', type=str, default='GeoSound_pcmepp_bingmap') #Options: #["GeoSound_pcmepp_bingmap","GeoSound_pcmepp_metadata_bingmap",
                                                                  #"GeoSound_pcmepp_sentinel","GeoSound_pcmepp_metadata_sentinel"]
    parser.add_argument('--samples',type=int, default=3) #number of samples to retrieve  
    args = parser.parse_args()
    

    ckpt_path = ckpt_cfg[args.expr]

    retrieval_engine = audio_retrieval_engine(args=args,ckpt_path=ckpt_path,audio_gallery_embeds_path=args.gallery_embeds_path,
                                              metadata_type=args.metadata_type,device=device)
    
    assert args.sat_query_path != "" or ("bingmap" in args.expr and args.sat_type == "bingmap") #this demo downloads image only from bingmap for now.
    
    if args.sat_query_path == "":
        #download image
        zoom_level = 18
        # image settings
        im_size = "1500,1500"
        template_url = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%%s/%d?mapSize=%s&key=%s" % (zoom_level, im_size, bingmap_api)
        tmp_loc = "%s,%s" % (args.lat, args.long)
        image_url = template_url % (tmp_loc)
        image_file = os.path.join(args.output_dir,'demo.jpeg')
        print(image_file)
        download(url=image_url,out_file=image_file)
        print("Satellite Image downloaded")
        sat_image_path = image_file
    else:
        sat_image_path = args.sat_query_path

    processed_sat_image = process_sat_image(sat_image_path=sat_image_path,sat_type=args.sat_type,
                                            zoom_level=args.zoom_level)
    
    processed_metadata = process_metadata(args)
    processed_metadata['sat'] = processed_sat_image

    top_retreived = retrieval_engine.get_topk_audios(processed_metadata)
    print(top_retreived)
    # exec(os.environ.get("DEBUG"))