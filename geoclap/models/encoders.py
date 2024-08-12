import torch
import pytorch_lightning as pl
from .CLAP import  CLAP_audiomodel_withProjection, CLAP_textmodel_withProjection
from .SATMAE import SatMAE_baseline, ProbSatMAE
import torch.nn as nn

                             
def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)


class Projector(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x

class sat_encoder(pl.LightningModule):
    def __init__(self, pretrained_model_path,feat_dim=768,fc_dim = 512,global_pool=False, metadata_type='none',meta_droprate=0.5,encoder_type='probSatMAE'):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'baselineSatMAE':
            self.encoder = SatMAE_baseline(pretrained_model_path=pretrained_model_path,feat_dim=feat_dim,fc_dim = fc_dim,global_pool=global_pool, metadata_type=metadata_type,meta_droprate=meta_droprate)
        
        elif encoder_type == 'probSatMAE':
            self.encoder = ProbSatMAE(pretrained_model_path=pretrained_model_path,feat_dim=feat_dim,fc_dim = fc_dim,global_pool=global_pool, metadata_type=metadata_type,meta_droprate=meta_droprate)
        else:
            raise NotImplementedError(f"allowed sat_encoder types are :[baselineSatMAE, probSatMAE] BUT provided: {encoder_type}")
        
    def forward(self, images, zoom_level=torch.tensor([]),sat_type="sentinel",audio_source=None, caption_source=None, latlong=None, time= None, month = None, time_valid=None, month_valid=None, eval_meta=None):
        output = self.encoder(x=images,zoom_level=zoom_level,sat_type=sat_type,audio_source=audio_source,caption_source=caption_source, latlong=latlong, time= time, month = month, time_valid=time_valid, month_valid=month_valid, eval_meta=eval_meta)
        return {'mean':l2normalize(output['mean']),'std':output['std'], 'unnormalized_mean':output['mean']}

class audio_encoder(pl.LightningModule):
    def __init__(self, feat_dim=512,fc_dim = 512,encoder_type='probCLAP', freeze = True):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'baselineCLAP': #For deterministic experiments, does not use GPO rather uses embedding from linear projection layer
            self.model = CLAP_audiomodel_withProjection(freeze=freeze)
            self.head = Projector(feat_dim, fc_dim)
        elif encoder_type == 'probCLAP': #For probabilistic experiments, does not use GPO, instead creates two simple MLP heads for mean and std head
            self.model = CLAP_audiomodel_withProjection(freeze=freeze)
            self.mean_head = Projector(feat_dim, fc_dim)
            self.std_head = Projector(feat_dim, fc_dim)
            self.std_head.linear1.bias.data.fill_(-4)#Set the bias of the linear layer to -4 to prevent too large std
            self.std_head.linear2.bias.data.fill_(-4)#Set the bias of the linear layer to -4 to prevent too large std
        else: 
            raise NotImplementedError(f"allowed audio_encoder types are :[baselineCLAP,  probCLAP] BUT provided: {encoder_type}")
             
    def forward(self,x):
        x = self.model(x)
        batch_embeddings_std = None
        if self.encoder_type == 'baselineCLAP':
            batch_embeddings = self.head(x)
            batch_embeddings_mean = batch_embeddings
        elif self.encoder_type == "probCLAP":
            batch_embeddings_mean = self.mean_head(x)
            batch_embeddings_std = self.std_head(x)
        else:
            raise NotImplementedError("allowed audio_encoder types are :[baselineCLAP, probCLAP]")
        
        return {'mean':l2normalize(batch_embeddings_mean),'std':batch_embeddings_std, 'unnormalized_mean':batch_embeddings_mean} 


class text_encoder(pl.LightningModule):
    def __init__(self, feat_dim=512,fc_dim = 512,encoder_type='probCLAP', freeze = True):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'baselineCLAP': #For deterministic experiments, does not use GPO rather uses embedding from linear projection layer
            self.model = CLAP_textmodel_withProjection(freeze=freeze)
            self.head = Projector(feat_dim, fc_dim)
     
        elif encoder_type == 'probCLAP': #For probabilistic experiments, does not use GPO, instead creates two simple MLP heads for mean and std head
            self.model = CLAP_textmodel_withProjection(freeze=freeze)
            self.mean_head = Projector(feat_dim, fc_dim)
            self.std_head = Projector(feat_dim, fc_dim)
            self.std_head.linear1.bias.data.fill_(-4)#Set the bias of the linear layer to -4 to prevent too large std
            self.std_head.linear2.bias.data.fill_(-4)

        else: 
            raise NotImplementedError(f"allowed text_encoder types are :[baselineCLAP, probCLAP] BUT provided: {encoder_type}")
        
    def forward(self,x):
        x = self.model(x)
        batch_embeddings_std = None
        if self.encoder_type == 'baselineCLAP':
            batch_embeddings = self.head(x)
            batch_embeddings_mean = batch_embeddings
        elif self.encoder_type == "probCLAP":
            batch_embeddings_mean = self.mean_head(x)
            batch_embeddings_std = self.std_head(x)
        else:
            raise NotImplementedError("allowed audio_encoder types are :[baselineCLAP, probCLAP]") 
        return {'mean':l2normalize(batch_embeddings_mean),'std':batch_embeddings_std, 'unnormalized_mean':batch_embeddings_mean}


if __name__ == '__main__':
    # some sanity check of code
    from ..utilities import clap_data_processor
    import os
    import torchaudio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_audio = '/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/raw_audio/aporee_9995_11936/EssenRathausPassageRolltreppen.mp3'
    track, sr = torchaudio.load(path_to_audio) 
    audio_sample =  clap_data_processor.get_audio_clap(track, sr)
                                                               # dict_keys(['input_features', 'is_longer'])
    audio_sample['input_features'] = audio_sample['input_features']

    text_sample = clap_data_processor.get_text_clap(['dummy text','dummy_text2'])

    sat_image_sample = torch.randn(5, 3, 224, 224).to(device)

    sat_encoders = ['baselineSatMAE', 'probSatMAE']
    clap_encoders = ['baselineCLAP', 'probCLAP']

    SatMAE_pretrained_model_path = "/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/SATMAE/pretrain-vit-base-e199.pth"

    for e in sat_encoders:
        print("Testing for:",e)
        pretrained_model_path = SatMAE_pretrained_model_path
        model = sat_encoder(pretrained_model_path=pretrained_model_path, encoder_type=e).to(device)
        output = model(sat_image_sample,sat_type='sentinel',zoom_level=torch.tensor([4,2,3,1,5]).to(device))
        print("Mean embed shape:",output['mean'].shape)
        if 'probSatMAE' in e:
            print("Std embed shape:",output['std'].shape)
    
    print("Test for TEXT encoders")
    for e in clap_encoders:
        print("Testing for:",e)
        model = text_encoder(encoder_type=e)
        output = model(text_sample)
        print("Mean embed shape:",output['mean'].shape)
        if 'probCLAP' in e:
            print("Std embed shape:",output['std'].shape)

    print("Test for AUDIO encoders")
    for e in clap_encoders:
        print("Testing for:",e)
        model = audio_encoder(encoder_type=e)
        output = model(audio_sample)
        print("Mean embed shape:",output['mean'].shape)
        if 'probCLAP' in e:
            print("Std embed shape:",output['std'].shape)



