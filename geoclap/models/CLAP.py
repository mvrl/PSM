#Hugging face way of loading CLAP model

from transformers import ClapAudioModelWithProjection, ClapTextModelWithProjection, ClapAudioModel, ClapTextModel
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange


class CLAP_audiomodel_withProjection(pl.LightningModule):
    def __init__(self,freeze=True):
        super().__init__()
        if freeze:
            self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").eval()
            for params in self.model.parameters():
                params.requires_grad=False
        else:
            self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").train()
    def forward(self,audio):
        batch_embeddings_audio = self.model(**audio)['audio_embeds']
        return batch_embeddings_audio


class CLAP_textmodel_withProjection(pl.LightningModule):
    def __init__(self,freeze=True):
        super().__init__()
        if freeze:
            self.model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused").eval()
            for params in self.model.parameters():
                params.requires_grad=False
        else:
            self.model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused").train()

    def forward(self,text):
        batch_embeddings_text = self.model(**text)['text_embeds']
        return batch_embeddings_text


class CLAP_audiomodel(pl.LightningModule):
    def __init__(self,freeze=True):
        super().__init__()
        if freeze:
            self.model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").eval()
            for params in self.model.parameters():
                params.requires_grad=False
        else:
            self.model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").train()
    def forward(self,audio):
        batch_embeddings_audio = self.model(**audio).last_hidden_state
        batch_embeddings_audio = rearrange(batch_embeddings_audio,'b d h w -> b (h w) d')
        return batch_embeddings_audio


class CLAP_textmodel(pl.LightningModule):
    def __init__(self,freeze=True):
        super().__init__()
        if freeze:
            self.model = ClapTextModel.from_pretrained("laion/clap-htsat-fused").eval()
            for params in self.model.parameters():
                params.requires_grad=False
        else:
            self.model = ClapTextModel.from_pretrained("laion/clap-htsat-fused").train()
        self.model.pooler = None #Pooling is not used in this scenario, we use GPO.
    def forward(self,text):
        batch_embeddings_text = self.model(**text).last_hidden_state
        return batch_embeddings_text


if __name__ == '__main__':
    from ..utilities import clap_data_processor
    import torchaudio
    path_to_audio = '/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee/raw_audio/aporee_9995_11936/EssenRathausPassageRolltreppen.mp3'
    track, sr = torchaudio.load(path_to_audio) 
    audio_sample =  clap_data_processor.get_audio_clap(track, sr)
    print(audio_sample.keys())                                                              # dict_keys(['input_features', 'is_longer'])
    audio_sample['input_features'] = audio_sample['input_features']
    print(audio_sample['input_features'].shape,audio_sample['is_longer'].shape)             # torch.Size([1, 4, 1001, 64]) torch.Size([1, 1])
    print(audio_sample['is_longer'])
    text_sample = clap_data_processor.get_text_clap(['dummy text','dummy_text2'])
    print(text_sample['attention_mask'].shape, text_sample['input_ids'].shape)              # torch.Size([2, 128]) torch.Size([2, 128])

    text_model1 = CLAP_textmodel_withProjection()
    print(text_model1(text_sample).shape)                                                   # torch.Size([2, 512])
    text_model2 = CLAP_textmodel()
    print(text_model2(text_sample).shape)                                                   # torch.Size([2, 128, 768])

    audio_model1 = CLAP_audiomodel_withProjection()
    print(audio_model1(audio_sample).shape)                                                 # torch.Size([1, 512])
    audio_model2 = CLAP_audiomodel()
    print(audio_model2(audio_sample).shape)                                                 # torch.Size([1, 64, 768])

