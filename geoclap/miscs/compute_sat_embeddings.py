# Given the checkpoint of a pretrained model, this script computes and saves overhead-imagery embeddings for all the images indexed in a region file csv.
from argparse import ArgumentParser, RawTextHelpFormatter
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from ..engine import GeoCLAPModel
from ..ckpt_paths import ckpt_cfg
from tqdm import tqdm
import pandas as pd
from PIL import Image
import random
from ..utilities.SatImage_transform import zoom_transform, sat_transform
from argparse import Namespace
import h5py as h5
import code

image_gsd = {'sentinel':10, 'bingmap':0.6, 'googleEarth':0.2}

def read_csv(csv_path,data_path, sat_type="sentinel"):
    region_file = pd.read_csv(os.path.join(csv_path))
    ids = list(region_file['id'])
    lats = list(region_file['Y'])
    longs = list(region_file['X'])
    if sat_type == "sentinel":
        paths = [os.path.join(data_path,str(id)+'.jpeg') for id in ids]
    elif sat_type == "bingmap": #file format : 1000000.0_39.274496967_-91.441179236.jpg
        ids_detail = [str(ids[i])+'.0_'+str(lats[i])+'_'+str(longs[i])+'.jpg' for i in range(len(region_file))]
        paths = [os.path.join(data_path,str(ids_detail[i])) for i in range(len(region_file))]
    else:
        raise NotImplementedError("allowed satellite imagery types are: sentinel and bingmap")
    return ids, paths, lats, longs

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

def move_to_device(item, device):
    if item[0]!= None:
        item = item.to(device)
    return item

class region_dataset(Dataset):
    def __init__(self, region_file_path,data_path, sat_type="sentinel"):
        super().__init__()
        self.sat_type = sat_type
        self.keys, self.img_paths, self.lats, self.longs = read_csv(region_file_path,data_path,sat_type)
        self.sat_transform = sat_transform(is_train=False, input_size=224)

    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        out = {'key':None,'sat':None, 'latitude':None,'longitude':None}
        sat_image = Image.open(os.path.join(self.img_paths[idx]))

        zoom_tr = zoom_transform(zoom_level=1, sat_type=self.sat_type)
        level_image1 = zoom_tr(sat_image)
        level_image1 = np.array(torch.permute(level_image1,[1,2,0]))
        final_image1 = self.sat_transform(level_image1)

        zoom_tr = zoom_transform(zoom_level=3, sat_type=self.sat_type)
        level_image3 = zoom_tr(sat_image)
        level_image3 = np.array(torch.permute(level_image3,[1,2,0]))
        final_image3 = self.sat_transform(level_image3)

        zoom_tr = zoom_transform(zoom_level=5, sat_type=self.sat_type)
        level_image5 = zoom_tr(sat_image)
        level_image5 = np.array(torch.permute(level_image5,[1,2,0]))
        final_image5 = self.sat_transform(level_image5)

        out['key'] = self.keys[idx]
        out['sat_1'] = final_image1
        out['sat_3'] = final_image3
        out['sat_5'] = final_image5
        out['latitude'] = self.lats[idx]
        out['longitude'] = self.longs[idx]
        
        return [out['key'], out['sat_1'], out['sat_3'], out['sat_5'], out['latitude'], out['longitude']]
    
class sat_embeds_engine(object):
    def __init__(self,ckpt_path,device):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = device
        self.model, self.hparams = self.get_geoclap()

    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        hparams["mode"] = "evaluate"
        hparams['metadata_type'] = "none"
        model = GeoCLAPModel(Namespace(**hparams)).to(self.device)
        model.load_state_dict(pretrained_weights,strict=False)
        geoclap = model.eval()
        #set all requires grad to false
        for params in geoclap.parameters():
            params.requires_grad=False
        model = geoclap
        
        return model, hparams  
    
    def get_batch(self,batch):
        #[out['key'], out['sat_1'], out['sat_3'], out['sat_5'], out['latitude'], out['longitude']]
        out_dict =    {'key':batch[0],
                      'sat_1':move_to_device(batch[1],self.device),
                      'sat_3':move_to_device(batch[2],self.device),
                      'sat_5':move_to_device(batch[3],self.device),
                      'latitude': batch[4],
                      'longitude': batch[5]
                      }
        return out_dict
    
    def get_sat_embeds(self,sat_image,zoom_level):
        zoom_levels = [zoom_level]*sat_image.shape[0]
        input_res = torch.tensor([1.0*z*image_gsd[self.hparams['sat_type']] for z in zoom_levels])
        
        sat_feats = self.model.sat_encoder.encoder.backbone(sat_image,input_res=input_res)
        sat_feats = self.model.sat_encoder.encoder.projector(sat_feats)
        
        sat_mean  = self.model.sat_encoder.encoder.mean_head(sat_feats)
        sat_std  = self.model.sat_encoder.encoder.std_head(sat_feats)

        return sat_feats.detach().cpu(), sat_mean.detach().cpu(), sat_std.detach().cpu()


    def get_batch_embeds(self,batch):
        batch = self.get_batch(batch)
        embeds = {'keys':batch['key'],
                  'lats':batch['latitude'],
                  'longs':batch['longitude'],
                  }
        embeds["sat_embeddings_1"], embeds["sat_embeddings_mean_1"], embeds["sat_embeddings_std_1"] = self.get_sat_embeds(batch["sat_1"],zoom_level=1)
        embeds["sat_embeddings_3"], embeds["sat_embeddings_mean_3"], embeds["sat_embeddings_std_3"] = self.get_sat_embeds(batch["sat_3"],zoom_level=3)
        embeds["sat_embeddings_5"], embeds["sat_embeddings_mean_5"], embeds["sat_embeddings_std_5"] = self.get_sat_embeds(batch["sat_5"],zoom_level=5)

        embeds['sat_std_norm_1'] = torch.sqrt(torch.exp(embeds["sat_embeddings_std_1"])).sum(dim=-1).numpy()
        embeds['sat_std_norm_3'] = torch.sqrt(torch.exp(embeds["sat_embeddings_std_3"])).sum(dim=-1).numpy()
        embeds['sat_std_norm_5'] = torch.sqrt(torch.exp(embeds["sat_embeddings_std_5"])).sum(dim=-1).numpy()
        
        return embeds
    
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_path', type=str, default="/storage1/fs1/jacobsn/Active/project_crossviewmap")
    parser.add_argument('--sat_type', type=str, default='sentinel', choices=["sentinel","bingmap"])
    parser.add_argument('--output_dir_path', type=str, default="/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/embeds")
    parser.add_argument('--info', type=str, default='USA embeddings with the best performing ckpt:')
    parser.add_argument('--expr', type=str, default='GeoSound_pcmepp_sentinel') #Options: #["GeoSound_pcmepp_bingmap","GeoSound_pcmepp_metadata_bingmap","GeoSound_pcmepp_sentinel","GeoSound_pcmepp_metadata_sentinel"]
    parser.add_argument('--embeds_size', type=int, default=512)

    args = parser.parse_args()
    
    expr_name = args.expr
    assert args.sat_type in expr_name
    ckpt_path = ckpt_cfg[expr_name]
    region_file = os.path.join(args.data_path,"USA_"+args.sat_type.upper(),"USA_6KM_grid_"+args.sat_type+"_clean.csv")
    sat_data_path = os.path.join(args.data_path,"USA_"+args.sat_type.upper(),"images")
    embed_compute_class = sat_embeds_engine(ckpt_path=ckpt_path,
                             device=device,
                             )
    
    output_name = "USA_embeds_with_"+expr_name
    output_size = len(pd.read_csv(region_file))
    predloader = DataLoader(region_dataset(region_file_path=region_file,data_path=sat_data_path,
                                               sat_type=args.sat_type),
                                            num_workers=8, batch_size=1000, shuffle=False, drop_last=False, pin_memory=True)
    
    output_file = f'{args.output_dir_path}/{output_name}.h5'
    
    with h5.File(output_file, 'w') as f:

        print(f'Creating new h5 file:{output_file}')
        dset_sat_embeddings_1 = f.create_dataset('sat_embeddings_1', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_embeddings_mean_1 = f.create_dataset('sat_embeddings_mean_1', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_embeddings_std_1 = f.create_dataset('sat_embeddings_std_1', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_std_norm_1 = f.create_dataset('sat_std_norm_1', shape=(output_size,), dtype=np.float32)

        dset_sat_embeddings_3 = f.create_dataset('sat_embeddings_3', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_embeddings_mean_3 = f.create_dataset('sat_embeddings_mean_3', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_embeddings_std_3 = f.create_dataset('sat_embeddings_std_3', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_std_norm_3 = f.create_dataset('sat_std_norm_3', shape=(output_size,), dtype=np.float32)

        dset_sat_embeddings_5 = f.create_dataset('sat_embeddings_5', shape=(output_size,args.embeds_size), dtype=np.float32)    
        dset_sat_embeddings_mean_5 = f.create_dataset('sat_embeddings_mean_5', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_embeddings_std_5 = f.create_dataset('sat_embeddings_std_5', shape=(output_size,args.embeds_size), dtype=np.float32)
        dset_sat_std_norm_5 = f.create_dataset('sat_std_norm_5', shape=(output_size,), dtype=np.float32)

        dset_location = f.create_dataset('location', shape=(output_size, 2), dtype=np.float32)
        dset_key = f.create_dataset('key', shape=(output_size,1), dtype=np.float32)
        
        f.attrs['model_path'] = ckpt_path
        f.attrs['sat_data_path'] = sat_data_path
        f.attrs['comment'] = args.info + ckpt_path

        print('Computing Embeddings')
        curr = 0
        for batch in tqdm(predloader):

            outputs = embed_compute_class.get_batch_embeds(batch)
            keys = outputs['keys']
            lats = outputs['lats']
            longs = outputs['longs']

            sat_embeddings_1 = outputs['sat_embeddings_1']
            sat_embeddings_mean_1 = outputs['sat_embeddings_mean_1']
            sat_embeddings_std_1 = outputs['sat_embeddings_std_1']
            sat_std_norm_1 =  outputs['sat_std_norm_1'] 

            sat_embeddings_3 = outputs['sat_embeddings_3']
            sat_embeddings_mean_3 = outputs['sat_embeddings_mean_3']
            sat_embeddings_std_3 = outputs['sat_embeddings_std_3']
            sat_std_norm_3 =  outputs['sat_std_norm_3'] 

            sat_embeddings_5 = outputs['sat_embeddings_5']
            sat_embeddings_mean_5 = outputs['sat_embeddings_mean_5']
            sat_embeddings_std_5 = outputs['sat_embeddings_std_5']
            sat_std_norm_5 =  outputs['sat_std_norm_5'] 

            
            batch_size = sat_embeddings_mean_1.shape[0]
            
            locations = np.array(torch.cat([lats.unsqueeze(1),longs.unsqueeze(1)],axis=1))
            keys = np.array(keys).reshape((-1,1)).astype(int)

            #add data to file
            dset_sat_embeddings_1[curr:curr+batch_size] = np.array(sat_embeddings_1)
            dset_sat_embeddings_mean_1[curr:curr+batch_size] = np.array(sat_embeddings_mean_1)
            dset_sat_embeddings_std_1[curr:curr+batch_size] = np.array(sat_embeddings_std_1)
            dset_sat_std_norm_1[curr:curr+batch_size] = np.array(sat_std_norm_1)
            
            dset_sat_embeddings_3[curr:curr+batch_size] = np.array(sat_embeddings_3)
            dset_sat_embeddings_mean_3[curr:curr+batch_size] = np.array(sat_embeddings_mean_3)
            dset_sat_embeddings_std_3[curr:curr+batch_size] = np.array(sat_embeddings_std_3)
            dset_sat_std_norm_3[curr:curr+batch_size] = np.array(sat_std_norm_3)

            dset_sat_embeddings_5[curr:curr+batch_size] = np.array(sat_embeddings_5)
            dset_sat_embeddings_mean_5[curr:curr+batch_size] = np.array(sat_embeddings_mean_5)
            dset_sat_embeddings_std_5[curr:curr+batch_size] = np.array(sat_embeddings_std_5)
            dset_sat_std_norm_5[curr:curr+batch_size] = np.array(sat_std_norm_5)

            dset_location[curr:curr+batch_size] = locations
            dset_key[curr:curr+batch_size] = keys

            curr += batch_size
            # code.interact(local=dict(globals(), **locals()))
            if curr >= output_size:
                print('Max size reached')
                break
                
    print(f'File saved to {output_file}')