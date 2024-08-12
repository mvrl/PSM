#This script computes embeddings for the test-set, gallery. It could be used for cross-modal retrieval demo later:
##local imports
from ..engine import GeoCLAPModel
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from ..dataloader import Dataset_soundscape
from argparse import Namespace, ArgumentParser, RawTextHelpFormatter
from ..config import cfg
from ..ckpt_paths import ckpt_cfg
import webdataset as wds
import pandas as pd
import json
import h5py as h5

DS_SIZE = {'test':10000,'val':5000}
image_gsd = {'sentinel':10, 'bingmap':0.6, 'googleEarth':0.2}

def get_shards(dataset_type="GeoSound",overhead_type="sentinel"):
    if dataset_type == "GeoSound":
        data_path = os.path.join(cfg.GeoSound_webdataset_path,"with_"+overhead_type)
    else:
        data_path = cfg.SoundingEarth_webdataset_path
    all_shards = [os.path.join(data_path,s) for s in os.listdir(data_path) if ".tar" in s]
    test_shard = [s for s in all_shards if 'test' in s]
    val_shard = [s for s in all_shards if 'val' in s]
    train_shards = [s for s in all_shards if 'train' in s]
    return val_shard, test_shard

def move_to_device(item, device):
    if item[0]!= None:
        item = item.to(device)
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


class Evaluate(object):
    def __init__(self, split, ckpt_path,device, metadata_type="none",meta_droprate = 0.0, add_text="false",test_zoom_level=1, test_mel_index=0, dataset_type="GeoSound", sat_type="sentinel", loss_type="pcmepp"):
        super().__init__()
        self.split = split
        self.loss_type = loss_type
        self.ckpt_path = ckpt_path
        self.device = device
        self.metadata_type = metadata_type
        self.meta_droprate = meta_droprate
        self.add_text = add_text
        self.test_zoom_level = test_zoom_level
        self.test_mel_index = test_mel_index
        self.dataset_type = dataset_type
        self.sat_type = sat_type
       
    def get_batch(self,batch):
        out_dict =    {'key':batch[0],
                      'audio_source':move_to_device(batch[1],self.device), 'caption_source':move_to_device(batch[2],self.device),
                      'audio':{'input_features':move_to_device(batch[3],self.device), 'is_longer':move_to_device(batch[4],self.device)}, 
                      'text':{'input_ids':move_to_device(batch[5],self.device),'attention_mask':move_to_device(batch[6],self.device)},
                      'sat_zoom_level':move_to_device(batch[7],self.device),'sat':move_to_device(batch[8],self.device),
                      'latlong':move_to_device(batch[9],self.device), 'time':move_to_device(batch[10],self.device), 'month':move_to_device(batch[11],self.device),
                      'time_valid':move_to_device(batch[12],self.device), 'month_valid':move_to_device(batch[13],self.device)} 
        return out_dict
    
    def get_embeds(self,batch):
        batch = self.get_batch(batch)
        embeds = {'sat_embeddings':None, 'audio_embeddings':None, 'text_embeddings':None}
        zoom_levels = [self.test_zoom_level]*batch['sat'].shape[0]
        input_res = torch.tensor([1.0*z*image_gsd[self.hparams.sat_type] for z in zoom_levels])
        
        sat_feats = self.model.sat_encoder.encoder.backbone(batch['sat'],input_res=input_res)
        sat_feats = self.model.sat_encoder.encoder.projector(sat_feats)
        embeds['sat_embeddings'] = sat_feats
        
        batch_audio = {}
        for key in batch['audio'].keys():
            batch_audio[key] = batch['audio'][key]
        embeds['audio_embeddings'] = self.model.audio_encoder(batch_audio)
        
        if self.hparams.modality_type == 'sat_audio_text':   
            batch_text = {}
            for key in batch['text'].keys():
                batch_text[key] = batch['text'][key]
            embeds['text_embeddings'] = self.model.text_encoder(batch_text)
        
        return embeds

    def get_dataloader(self,hparams):
        hparams['test_zoom_level'] = self.test_zoom_level
        hparams['test_mel_index'] = self.test_mel_index
        hparams['test_batch_size'] = 500
        hparams['dataset_type'] = self.dataset_type
        hparams = Namespace(**hparams)
        testset = Dataset_soundscape(hparams,is_train=False,test_zoom_level=self.test_zoom_level).get_ds(mode=self.split)
        
        testloader = wds.WebLoader(testset, batch_size=None,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=8)
        self.hparams = hparams
        return testloader
    
    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        assert (hparams['dataset_type'] == self.dataset_type) and (hparams['loss_type'] == self.loss_type) and  (hparams['sat_type'] == self.sat_type)#just a safety check to ensure usage of right checkpoint
        pretrained_weights = pretrained_ckpt['state_dict']
        val_shard, test_shard = get_shards(dataset_type="GeoSound",overhead_type=hparams['sat_type'])
        if self.split == "test":
            hparams['test_path'] = test_shard
        else:
            hparams['test_path'] = val_shard
        hparams["mode"] = "evaluate"
        hparams['meta_droprate'] = self.meta_droprate
        model = GeoCLAPModel(Namespace(**hparams)).to(self.device)
        model.load_state_dict(pretrained_weights,strict=False)
        geoclap = model.eval()
        #set all requires grad to false
        for params in geoclap.parameters():
            params.requires_grad=False
        self.model = geoclap
        
        return hparams  
    
    @torch.no_grad()
    def get_final_metrics(self):
        set_seed(56)
        hparams  = self.get_geoclap()
        test_dataloader = self.get_dataloader(hparams)
        print(hparams)
        sat_embeddings = []
        audio_embeddings_mean = []
        text_embeddings_mean = []
        text_embeddings_std = []
        audio_embeddings_std = []
        keys = []
        for i,batch in tqdm(enumerate(test_dataloader)):
            if len(set(keys)) == DS_SIZE[self.split]:
                break
            else:
                print("batch no:",str(i))
                keys = keys + list(batch[0])
                embeds = self.get_embeds(batch=batch)
                sat_embeddings.append(embeds['sat_embeddings'])
                audio_embeddings_mean.append(embeds['audio_embeddings']['unnormalized_mean'])
                
                if "text" in self.hparams.modality_type:
                    text_embeddings_mean.append(embeds['text_embeddings']['unnormalized_mean'])
                
                if hparams['probabilistic']:
                    audio_embeddings_std.append(embeds['audio_embeddings']['std'])
                    if "text" in self.hparams.modality_type:
                        text_embeddings_std.append(embeds['text_embeddings']['std'])
            
        sat_embeddings = torch.cat(sat_embeddings,axis=0).to(self.device)
        audio_embeddings_mean = torch.cat(audio_embeddings_mean,axis=0).to(self.device)
        if "text" in self.hparams.modality_type:
            text_embeddings_mean = torch.cat(text_embeddings_mean,axis=0).to(self.device)

        if hparams['probabilistic']:
            audio_embeddings_std = torch.cat(audio_embeddings_std,axis=0).to(self.device)
            if "text" in self.hparams.modality_type:
                text_embeddings_std = torch.cat(text_embeddings_std,axis=0).to(self.device)
        
        sat_embeddings_dict = {'sat_feat':sat_embeddings}
        audio_embeddings_dict = {'mean':audio_embeddings_mean,'std':audio_embeddings_std}
        if "text" in self.hparams.modality_type:
            text_embeddings_dict = {'mean':text_embeddings_mean,'std':text_embeddings_std}
        print(sat_embeddings.shape)
        
        print("TEST if all keys are unique in test set?",len(keys)==len(set(keys)))

        return keys, sat_embeddings_dict, audio_embeddings_dict, text_embeddings_dict
        

#GeoSound_infonce_sentinel
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--results_path', type=str, default='/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results')
    parser.add_argument('--test_zoom_level', type=int, default=1)
    parser.add_argument('--test_mel_index', type=int, default=0,choices=[0,1,2,3,4])
    parser.add_argument('--meta_droprate', type=float, default=1.0) # if 1.0: no metadata will be kept, if 0.0: all metadata will be kept
    parser.add_argument('--split', type=str, default="test") #options: val, test
    parser.add_argument('--dataset_type', type=str, default="GeoSound")
    parser.add_argument('--loss_type', type=str, default="infonce",choices=["pcmepp","infonce"])
    parser.add_argument('--sat_type', type=str, default='sentinel', choices=['sentinel','bingmap','googleEarth']) 
    parser.add_argument('--metadata_type', type=str, default='none',help="'latlong', 'month', 'time', 'asource', 'tsource','latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource', 'none'")
    parser.add_argument('--expr', type=str, default='') #Options: #["GeoSound_infonce_bingmap","GeoSound_infonce_metadata_bingmap","GeoSound_pcmepp_bingmap","GeoSound_pcmepp_metadata_bingmap",
                                                                  #"GeoSound_infonce_sentinel","GeoSound_infonce_metadata_sentinel","GeoSound_pcmepp_sentinel","GeoSound_pcmepp_metadata_sentinel"]
                                                        
                                                               
    args = parser.parse_args()
    assert (len(args.expr) !=0) or (len(args.ckpt_path) !=0) 
    #params
    set_seed(56)
    if args.ckpt_path != '':
        ckpt_path = args.ckpt_path
    else:
        #GeoSound_pcmepp_metadata_sentinel
        ckpt_path = ckpt_cfg[args.expr]

    #configure evaluation
    evaluation = Evaluate(split=args.split, ckpt_path=ckpt_path,device=device, 
                          metadata_type = args.metadata_type, meta_droprate=float(args.meta_droprate),
                          test_zoom_level=int(args.test_zoom_level), dataset_type=args.dataset_type, 
                          sat_type=args.sat_type, loss_type=args.loss_type)

    keys, sat_embeddings_dict, audio_embeddings_dict, text_embeddings_dict = evaluation.get_final_metrics()
    
    save_file = os.path.join(args.results_path,args.expr+ "_zoom_level_"+str(args.test_zoom_level)+"_gallery.h5")
               
    output_size = sat_embeddings_dict['sat_feat'].shape[0]
    embeds_size = 512

    with h5.File(save_file, 'w') as f:
        f.create_dataset('sat_feat', data=sat_embeddings_dict['sat_feat'].detach().cpu().numpy(),shape=(output_size,embeds_size), dtype=np.float32)

        f.create_dataset('audio_embeddings_mean', data=audio_embeddings_dict['mean'].detach().cpu().numpy(),shape=(output_size,embeds_size), dtype=np.float32)
        f.create_dataset('audio_embeddings_std', data=audio_embeddings_dict['std'].detach().cpu().numpy(),shape=(output_size,embeds_size), dtype=np.float32)  

        f.create_dataset('text_embeddings_mean', data=text_embeddings_dict['mean'].detach().cpu().numpy(),shape=(output_size,embeds_size), dtype=np.float32)
        f.create_dataset('text_embeddings_std', data=text_embeddings_dict['std'].detach().cpu().numpy(),shape=(output_size,embeds_size), dtype=np.float32)  

        f.create_dataset("key",data=keys)
        
        f.attrs['model_path'] = ckpt_path
        f.attrs['mel_index'] = args.test_mel_index
        f.attrs['dataset_type']=args.dataset_type
        f.attrs['overhead_type']=args.sat_type
        f.attrs['loss_type']=args.loss_type
        f.attrs['metadata_type']=args.metadata_type
        f.attrs['meta_droprate']=args.meta_droprate
        f.attrs['test_zoom_level']=args.test_zoom_level

    print("Saved gallery at:",save_file)
    # import code; code.interact(local=dict(globals(), **locals()))