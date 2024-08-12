#This script has original pretrained models for audio and satellite imagery modalities.
# These models will be used to compute cosine similarity between the ground-truth vs. top-k retrieved samples by our model.

from torchvision.io import read_image
from ..utilities import clap_data_processor, AST_audio
import torchaudio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from geoclap.utilities.SatImage_transform import sat_transform, zoom_transform
from transformers import ClapAudioModelWithProjection
from .SATMAE import get_SatMAE_model
from .SCALEMAE import get_ScaleMAE_model
from transformers import ASTModel
import pytorch_lightning as pl
import numpy as np
import torch

MODEL_FILE = '/storage1/fs1/jacobsn/Active/user_k.subash/projects/AudioCLIP/assets/AudioCLIP-Full-Training.pt'

image_gsd = {'sentinel':10, 'bingmap':0.6}
class SatMAE_eval(pl.LightningModule):
    def __init__(self, pretrained_model_path,device,global_pool=False):
        super().__init__()
        self.model = get_SatMAE_model(pretrained_model_path,device,global_pool).eval()
   
    def forward(self,x):
        x = self.model(x) 
        return x

class ScaleMAE_eval(pl.LightningModule):
    def __init__(self, pretrained_model_path,device,global_pool=False):
        super().__init__()
        self.model = get_ScaleMAE_model(finetune_ckpt=pretrained_model_path,device=device,global_pool= global_pool,cls_token = True).eval()
   
    def forward(self,x,sat_type, zoom_level):
        input_res = torch.tensor([1.0*z*image_gsd[sat_type] for z in zoom_level])
        x = self.model(x,input_res=input_res)
        return x

class AST_eval(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()
       
    def forward(self,audio):
        x = self.model(input_values = audio)
        return x

class CLAP_eval(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").eval()
    def forward(self,audio):
        x = self.model(**audio)['audio_embeds']
        return x


def get_processed_image(image_path,zoom_level,sat_type, device):
    zoom_tr = zoom_transform(zoom_level=zoom_level, sat_type=sat_type)
    sat_tr = sat_transform(is_train=False, input_size=224)
    #read_image
    sat_img = read_image(image_path)
    sat_img = np.array(torch.permute(sat_img,[1,2,0]))
    # print("Original image",sat_img.shape)

    #get image for a given zoom level
    level_image = zoom_tr(sat_img)
    # print("Image after zoom transform", level_image.shape)
    level_image = np.array(torch.permute(level_image,[1,2,0]))
    # print("Image after zoom transform", level_image.shape)

    #get image for resized one
    final_image = sat_tr(level_image).unsqueeze(0).to(device)
    # print("Image after sat transform",final_image.shape)
    return final_image

if __name__ == '__main__':
    # some sanity check of code

    ## TEST for sat images
    sat_image_path = '/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/images/bingmap'
    sat_type = "bingmap"
    satmae_pretrained_model_path = os.path.join('/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/','SATMAE','finetune-vit-base-e7.pth')
    scalemae_pretrained_model_path = "/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/scaleMAE/scalemae-vitlarge-800.pth"
    demo_image  =  os.listdir(sat_image_path)[0]
    zoom_level = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = os.path.join(sat_image_path,demo_image)

    final_image = get_processed_image(image_path,zoom_level,sat_type, device)

    satmae_model = SatMAE_eval(pretrained_model_path=satmae_pretrained_model_path,global_pool=False,device=device).to(device).eval()
    scalemae_model = ScaleMAE_eval(pretrained_model_path=scalemae_pretrained_model_path,device=device,global_pool=False).to(device).eval()
    print("sat_embeddings dim")
    print("for SATMAE:",satmae_model(final_image).shape)
    print("for SCALEMAE:",scalemae_model(final_image,sat_type="sentinel", zoom_level=[zoom_level]).shape)

    ## TEST for audio
    path_to_audio = '/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/raw_audio/aporee_9995_11936/EssenRathausPassageRolltreppen.mp3'
    track, sr = torchaudio.load(path_to_audio)
    track = track[:,:sr*10]  
    audio_sample =  clap_data_processor.get_audio_clap(track, sr)
    audio_sample['input_features'] = audio_sample['input_features'].unsqueeze(0)
    print("audio_embeddings dim")
    clap_audio_model = CLAP_eval()
    print("for CLAP:",clap_audio_model(audio_sample).shape)

    ast_audio_model = AST_eval()
    audio_sample = AST_audio.get_audio_AST(track, sr)
    audio_sample = audio_sample.unsqueeze(0)
    ast_feat = ast_audio_model(audio_sample)
    print("for AST:",ast_feat['pooler_output'].shape)
    # import code; code.interact(local=dict(globals(), **locals()))

