# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import numpy as np
import torch
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
import os
import pytorch_lightning as pl
from .pos_embed import get_2d_sincos_pos_embed_with_resolution
from .metadata_encoder import MetaTransformerEncoder

image_gsd = {'sentinel':10, 'bingmap':0.6, 'googleEarth':0.2}

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
            del self.cls_token
            self.class_token_flag = False
        else:
            self.class_token_flag = True

        del self.pos_embed
        del self.head

    def forward_features(self, x,input_res=None):
        B, _, h, w = x.shape
        input_res = input_res.cpu()
        x = self.patch_embed(x)
        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=self.class_token_flag,
            device=x.device,
        )
        if self.class_token_flag:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x.mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            print("returning cls token")
            outcome = x[:,0] #return cls token
        return outcome
    
    def forward(self, x, input_res=None):
        x = self.forward_features(x, input_res)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def get_SatMAE_model(ckpt_path, global_pool=False):

    checkpoint = torch.load(ckpt_path)
    model = vit_base_patch16(global_pool=global_pool)
    state_dict = model.state_dict()
    checkpoint_model = checkpoint['model']
    
    # for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    print(set(msg.missing_keys))
    # trunc_normal_(model.head.weight, std=2e-5)
    model_without_ddp = model
    # print(model_without_ddp)
    return model

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
###################################################################################################################
## Old way of fusing metadata using late-fusion strategy
# from .MLP import source_encoding, metaNet
# import random
# AUDIO_SOURCES = 4
# CAPTION_SOURCES = 3 #QWEN, PENGI, METADATA

# ## One way to implement modality dropout:
# def dropout(droprate=0.5):
#     percent = droprate *100
#     return random.randrange(0, 100) < percent

# class metadata_encoder(pl.LightningModule):
#     def __init__(self,metadata_type,meta_droprate=0.5, fc_dim=512):
#         super().__init__()
#         self.metadata_type = metadata_type
#         self.meta_droprate = meta_droprate
#         if 'asource' in self.metadata_type:
#             self.audio_source_fc = source_encoding(sources=AUDIO_SOURCES,fc_dim=fc_dim)
#         if 'tsource' in self.metadata_type:
#             self.caption_source_fc = source_encoding(sources=CAPTION_SOURCES,fc_dim=fc_dim)
#         if 'latlong' in self.metadata_type:
#             self.location_fc = metaNet(metadata_type="latlong",fc_dim = fc_dim)
#         if 'month' in self.metadata_type:
#             self.month_fc = metaNet(metadata_type="month",fc_dim = fc_dim)
#         if 'time' in self.metadata_type:
#             self.time_fc = metaNet(metadata_type="time",fc_dim = fc_dim)

#     def forward(self,sat_embeddings,audio_source=None,caption_source=None,latlong=None,month=None, time=None,time_valid=None, month_valid=None):

#         if 'asource' in self.metadata_type:
#             audio_source_embeddings = self.audio_source_fc(audio_source)
#             if dropout(self.meta_droprate):
#                 audio_source_embeddings = audio_source_embeddings*0
#             sat_embeddings = sat_embeddings + audio_source_embeddings
        
#         if 'tsource' in self.metadata_type:
#             caption_source_embeddings = self.caption_source_fc(caption_source)
#             if dropout(self.meta_droprate):
#                 caption_source_embeddings = caption_source_embeddings*0
#             sat_embeddings = sat_embeddings + caption_source_embeddings

#         if 'latlong' in self.metadata_type:
#             location_embeddings = self.location_fc(latlong)
#             if dropout(self.meta_droprate):
#                 location_embeddings = location_embeddings*0
#             sat_embeddings = sat_embeddings + location_embeddings
        
#         if 'month' in self.metadata_type:
#             month_embeddings = self.month_fc(month)
#             month_embeddings = month_embeddings*month_valid.unsqueeze(1)
#             if dropout(self.meta_droprate):
#                 month_embeddings = month_embeddings*0
#             sat_embeddings = sat_embeddings + month_embeddings
        
#         if 'time' in self.metadata_type:
#             time_embeddings = self.time_fc(time)
#             time_embeddings = time_embeddings*time_valid.unsqueeze(1)
#             if dropout(self.meta_droprate):
#                 time_embeddings = time_embeddings*0
#             sat_embeddings = sat_embeddings + time_embeddings

#         return sat_embeddings

################################################################################################################################

class SatMAE_baseline(pl.LightningModule):
    def __init__(self, pretrained_model_path,feat_dim=768,fc_dim = 512,global_pool=False, metadata_type='none',meta_droprate=0.5):
        super().__init__()
        self.metadata_type = metadata_type
        self.meta_droprate = meta_droprate
        self.backbone = get_SatMAE_model(ckpt_path=pretrained_model_path,global_pool=global_pool)
        self.projector = Projector(input_size=feat_dim,hidden_size=fc_dim)  
        if self.metadata_type != "none":
            self.meta_fuser = MetaTransformerEncoder(metadata_type=self.metadata_type,meta_droprate=meta_droprate,fc_dim=fc_dim)

    def forward(self,x,zoom_level,sat_type="bingmap",audio_source=None, caption_source=None, latlong=None, time= None, month = None, time_valid=None, month_valid=None, eval_meta=None):
        input_res = torch.tensor([1.0*z*image_gsd[sat_type] for z in zoom_level])
        x = self.backbone(x,input_res=input_res)
        sat_embeddings = self.projector(x)
        if self.metadata_type != "none":
            sat_embeddings = self.meta_fuser(sat_embeddings=sat_embeddings,audio_source=audio_source,caption_source=caption_source,latlong=latlong,month=month, time=time,time_valid=time_valid, month_valid=month_valid,eval_meta=eval_meta)
        return {'mean':sat_embeddings,'std':None} 


class ProbSatMAE(pl.LightningModule):
    def __init__(self, pretrained_model_path,feat_dim=768,fc_dim = 512,global_pool=False, metadata_type='none',meta_droprate=0.5):
        super().__init__()
        self.metadata_type = metadata_type
        self.meta_droprate = meta_droprate
        self.backbone = get_SatMAE_model(ckpt_path=pretrained_model_path,global_pool=global_pool)
        self.projector = Projector(input_size=feat_dim,hidden_size=fc_dim)
        if self.metadata_type != "none":
            self.meta_fuser = MetaTransformerEncoder(metadata_type=self.metadata_type,meta_droprate=meta_droprate,fc_dim=fc_dim)
        
        self.mean_head = Projector(input_size=fc_dim,hidden_size=fc_dim)
        self.std_head = Projector(input_size=fc_dim,hidden_size=fc_dim)
        self.std_head.linear1.bias.data.fill_(-4)#Set the bias of the linear layer to -4 to prevent too large std
        self.std_head.linear2.bias.data.fill_(-4)

    def forward(self,x,zoom_level,sat_type="bingmap",audio_source=None, caption_source=None, latlong=None, time= None, month = None, time_valid=None, month_valid=None,eval_meta=None):
        input_res = torch.tensor([1.0*z*image_gsd[sat_type] for z in zoom_level])
        x = self.backbone(x,input_res=input_res)
        sat_embeddings = self.projector(x)
        if self.metadata_type != "none":
            sat_embeddings = self.meta_fuser(sat_embeddings=sat_embeddings,audio_source=audio_source,caption_source=caption_source,latlong=latlong,month=month, time=time,time_valid=time_valid, month_valid=month_valid, eval_meta=eval_meta)

        sat_embeddings_mean = self.mean_head(sat_embeddings)
        sat_embeddings_std = self.std_head(sat_embeddings)

        return {'mean':sat_embeddings_mean,'std':sat_embeddings_std} 

if __name__ == '__main__':
    from ..utilities.SatImage_transform import sat_transform, zoom_transform
    from torchvision.io import read_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## MODEL SPECIFIC CODE
    encoder_type = "SATMAE"
    ckpt_path = "/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/SATMAE/pretrain-vit-base-e199.pth"
    model1 = SatMAE_baseline(ckpt_path,global_pool=False)
    model1.eval()
    model1 = model1.to(device)

    model2 = ProbSatMAE(ckpt_path)
    model2.eval()
    model2 = model2.to(device)
    ## MODEL SPECIFIC CODE

    sat_type = "bingmap"
    sat_image_path = '/storage1/fs1/jacobsn/Active/user_k.subash/data_archive/aporee/images/'+sat_type
    demo_image  =  os.listdir(sat_image_path)[0]
    zoom_level = 1
    global_pool = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #define transforms
    zoom_tr = zoom_transform(zoom_level=zoom_level, sat_type=sat_type)
    sat_tr = sat_transform(is_train=True, input_size=224)
    
    #read_image
    sat_img = read_image(os.path.join(sat_image_path,demo_image))
    sat_img = np.array(torch.permute(sat_img,[1,2,0]))
    print("Original image",sat_img.shape)

    #get image for a given zoom level
    level_image = zoom_tr(sat_img)
    print("Image after zoom transform", level_image.shape)
    level_image = np.array(torch.permute(level_image,[1,2,0]))
    print("Image after zoom transform", level_image.shape)

    #get image for resized one
    final_image = sat_tr(level_image).unsqueeze(0).to(device)
    print("Image after sat transform",final_image.shape)
    embeds = model1(final_image,[zoom_level],sat_type=sat_type)
    print(embeds["mean"].shape)

    embeds = model2(final_image,[zoom_level],sat_type=sat_type)
    print(embeds["mean"].shape, embeds["std"].shape)