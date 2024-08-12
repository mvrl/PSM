##local imports
from .metrics import get_retrevial
from .engine import GeoCLAPModel
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from .dataloader import Dataset_soundscape
from argparse import Namespace, ArgumentParser, RawTextHelpFormatter
from .config import cfg
from .ckpt_paths import ckpt_cfg
import webdataset as wds
import pandas as pd
import json


def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

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
    def __init__(self, split, ckpt_path,device, metadata_type="none",meta_droprate = 0.0, add_text="false",test_zoom_level=1, test_mel_index=0, recall_at=10,dataset_type="GeoSound", sat_type="sentinel", loss_type="pcmepp"):
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
        self.recall_at = recall_at
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
        if self.hparams.metadata_type != 'none':
            embeds['sat_embeddings']  = self.model.sat_encoder(batch['sat'],sat_type =self.hparams.sat_type, zoom_level = batch['sat_zoom_level'],
                                                        audio_source=batch['audio_source'], caption_source=batch['caption_source'],
                                                        latlong=batch['latlong'], time=batch['time'], month=batch['month'], 
                                                         time_valid=batch['time_valid'], month_valid=batch['month_valid'], eval_meta=self.metadata_type)
        else:
            embeds['sat_embeddings']  = self.model.sat_encoder(batch['sat'],sat_type =self.hparams.sat_type, zoom_level = batch['sat_zoom_level'])
        
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
        hparams[self.split+'_batch_size'] = 500
        hparams['dataset_type'] = self.dataset_type
        if 'caption_strategy' != hparams:
            hparams['caption_strategy'] = "original" 
        hparams = Namespace(**hparams)
        testset = Dataset_soundscape(hparams,is_train=False,test_zoom_level=self.test_zoom_level).get_ds(mode=self.split)
        
        testloader = wds.WebLoader(testset, batch_size=None,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=1)
        self.hparams = hparams
        return testloader
    
    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        assert (hparams['dataset_type'] == self.dataset_type) and (hparams['loss_type'] == self.loss_type) and  (hparams['sat_type'] == self.sat_type)#just a safety check to ensure usage of right checkpoint
        pretrained_weights = pretrained_ckpt['state_dict']
        val_shard, test_shard = get_shards(dataset_type=self.dataset_type,overhead_type=hparams['sat_type'])
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
        sat_embeddings_mean = []
        audio_embeddings_mean = []
        text_embeddings_mean = []
        text_embeddings_std = []
        sat_embeddings_std = []
        audio_embeddings_std = []
        keys = []
        for i,batch in tqdm(enumerate(test_dataloader)):
           
            print("batch no:",str(i))
            keys = keys + list(batch[0])
            embeds = self.get_embeds(batch=batch)
            sat_embeddings_mean.append(embeds['sat_embeddings']['unnormalized_mean'])
            audio_embeddings_mean.append(embeds['audio_embeddings']['unnormalized_mean'])
            
            if "text" in self.hparams.modality_type:
                text_embeddings_mean.append(embeds['text_embeddings']['unnormalized_mean'])
            
            if hparams['probabilistic']:
                sat_embeddings_std.append(embeds['sat_embeddings']['std'])
                audio_embeddings_std.append(embeds['audio_embeddings']['std'])
                if "text" in self.hparams.modality_type:
                    text_embeddings_std.append(embeds['text_embeddings']['std'])
            
        sat_embeddings_mean = torch.cat(sat_embeddings_mean,axis=0).to(self.device)
        audio_embeddings_mean = torch.cat(audio_embeddings_mean,axis=0).to(self.device)

        if "text" in self.hparams.modality_type:
            text_embeddings_mean = torch.cat(text_embeddings_mean,axis=0).to(self.device)

        if hparams['probabilistic']:
            sat_embeddings_std = torch.cat(sat_embeddings_std,axis=0).to(self.device)
            audio_embeddings_std = torch.cat(audio_embeddings_std,axis=0).to(self.device)
            if "text" in self.hparams.modality_type:
                text_embeddings_std = torch.cat(text_embeddings_std,axis=0).to(self.device)
        
        sat_embeddings_dict = {'mean':sat_embeddings_mean,'std':sat_embeddings_std}
        audio_embeddings_dict = {'mean':audio_embeddings_mean,'std':audio_embeddings_std}
        if "text" in self.hparams.modality_type:
            text_embeddings_dict = {'mean':text_embeddings_mean,'std':text_embeddings_std}
        print(sat_embeddings_mean.shape)
        
        R_k = self.recall_at/100*sat_embeddings_mean.shape[0] # Evaluation with Recall@R%
        print("TEST if all keys are unique in test set?",len(keys)==len(set(keys)))
        print(f'Recall@{R_k} for test gallery of size: {sat_embeddings_mean.shape[0]}')
        
        if hparams['probabilistic']:
            if self.add_text == "true":
                sat_query_embed_dict = {'mean':l2normalize(sat_embeddings_dict['mean']+text_embeddings_dict['mean']),'std':sat_embeddings_dict['std']+text_embeddings_dict['std']}
                audio_query_embed_dict = {'mean':l2normalize(audio_embeddings_dict['mean']+text_embeddings_dict['mean']),'std':audio_embeddings_dict['std']+text_embeddings_dict['std']}
            else:
                sat_query_embed_dict = {'mean':l2normalize(sat_embeddings_dict['mean']),'std':sat_embeddings_dict['std']}
                audio_query_embed_dict = {'mean':l2normalize(audio_embeddings_dict['mean']),'std':audio_embeddings_dict['std']}
        
        else:
            if self.add_text == "true":
                sat_query_embed_dict = {'mean':l2normalize(sat_embeddings_dict['mean']+text_embeddings_dict['mean'])}
                audio_query_embed_dict = {'mean':l2normalize(audio_embeddings_dict['mean']+text_embeddings_dict['mean'])}
            else:
                sat_query_embed_dict = {'mean':l2normalize(sat_embeddings_dict['mean'])}
                audio_query_embed_dict = {'mean':l2normalize(audio_embeddings_dict['mean'])}
        
        ## Image-to-Sound retrevial results:
        audio_embeddings_dict['mean'] = l2normalize(audio_embeddings_dict['mean'])
        retrieval_results_I2S, I2S_df = get_retrevial(modality1_emb=sat_query_embed_dict, modality2_emb=audio_embeddings_dict,keys=keys, normalized=True,k=R_k, loss_type=self.hparams.loss_type, save_top=10)

        sat_embeddings_dict['mean'] = l2normalize(sat_embeddings_dict['mean'])
        ## Sound-to-Image retrevial results:
        retrieval_results_S2I, S2I_df = get_retrevial(modality1_emb=audio_query_embed_dict, modality2_emb=sat_embeddings_dict, keys=keys, normalized=True,k=R_k, loss_type=self.hparams.loss_type, save_top=10)
        return retrieval_results_I2S, retrieval_results_S2I, I2S_df, S2I_df, R_k

#GeoSound_infonce_sentinel
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--results_path', type=str, default='/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results')
    parser.add_argument('--test_zoom_level', type=int, default=1)
    parser.add_argument('--test_mel_index', type=int, default=0,choices=[0,1,2,3,4])
    parser.add_argument('--meta_droprate', type=float, default=0.0) # if 1.0: no metadata will be kept, if 0.0: all metadata will be kept
    parser.add_argument('--add_text', type=str, default="false")
    parser.add_argument('--recall_at', type=int, default=10)
    parser.add_argument('--split', type=str, default="test") #options: val, test
    parser.add_argument('--dataset_type', type=str, default="GeoSound",choices=["GeoSound","SoundingEarth"])
    parser.add_argument('--loss_type', type=str, default="infonce",choices=["pcmepp","infonce"])
    parser.add_argument('--sat_type', type=str, default='sentinel', choices=['sentinel','bingmap','googleEarth']) 
    parser.add_argument('--save_results', type=str, default='false', choices=['true','false'])
    parser.add_argument('--json_name', type=str, default='main', help="'main','ablation','SoundingEarth','test"  )
    parser.add_argument('--metadata_type', type=str, default='none',help="'latlong', 'month', 'time', 'asource', 'tsource','latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource', 'none'")
    parser.add_argument('--expr', type=str, default='') #Options: #["GeoSound_infonce_bingmap","GeoSound_infonce_metadata_bingmap","GeoSound_pcmepp_bingmap","GeoSound_pcmepp_metadata_bingmap",
                                                                  #"GeoSound_infonce_sentinel","GeoSound_infonce_metadata_sentinel","GeoSound_pcmepp_sentinel","GeoSound_pcmepp_metadata_sentinel",
                                                                  #"SoundingEarth_infonce_googleEarth","SoundingEarth_infonce_metadata_googleEarth","SoundingEarth_pcmepp_googleEarth","SoundingEarth_pcmepp_metadata_googleEarth"]
                                                        
                                                               
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
                          metadata_type = args.metadata_type, meta_droprate=float(args.meta_droprate), add_text=args.add_text, recall_at = int(args.recall_at),
                          test_zoom_level=int(args.test_zoom_level), dataset_type=args.dataset_type, sat_type=args.sat_type, loss_type=args.loss_type)

    results_i2s, results_s2i, i2s_df, s2i_df, R_k = evaluation.get_final_metrics()
    print("IMAGE TO SOUND RETREVIAL RESULTS:",results_i2s)
    print("SOUND TO IMAGE RETREVIAL RESULTS:",results_s2i)
    
    results_dict = {'index':args.test_mel_index,
                    'dataset_type':args.dataset_type, 'overhead_type':args.sat_type, 'loss_type':args.loss_type,
                    'add_text':args.add_text, 'metadata_type':args.metadata_type, 'meta_droprate': args.meta_droprate,
                    'test_zoom_level':args.test_zoom_level, 'test_mel_index':args.test_mel_index,
                    'I2S_R@10':results_i2s['R@'+str(R_k)],'I2S_median':results_i2s['Median Rank'],
                    'S2I_R@10':results_s2i['R@'+str(R_k)],'S2I_median':results_s2i['Median Rank'],
                    'ckpt_path':ckpt_path
                    }
    
    json_path = ckpt_cfg["results_json"].replace(".json","_"+args.json_name+".json")
    save_dict_to_json(results_dict, json_path)
    if args.save_results == "true":
        s2i_df.to_csv(os.path.join(args.results_path,"results-s2i-"+args.expr+"-"+args.metadata_type+"-"+str(args.meta_droprate)+"-ZL-"+str(args.test_zoom_level)+"-"+str(args.test_mel_index)+".csv"))
        i2s_df.to_csv(os.path.join(args.results_path,"results-i2s-"+args.expr+"-"+args.metadata_type+"-"+str(args.meta_droprate)+"-ZL-"+str(args.test_zoom_level)+"-"+str(args.test_mel_index)+".csv"))
    # import code; code.interact(local=dict(globals(), **locals()))