import pytorch_lightning as pl
from itertools import chain
import os
from .config import cfg
from .models import encoders
from .criterions import get_criterion
from .criterions.loss_configs import get_loss_config
from .metrics import get_retrevial_metrics
from .dataloader import Dataset_soundscape
from .utilities.dist import grad_all_gather
import torch.distributed as dist
import torch
import webdataset as wds
import code


class GeoCLAPModel(pl.LightningModule):
    def __init__(self, hparams):

        #save paramaters
        super().__init__()
        #save initialized hyperparameters
        self.save_hyperparameters(hparams)
        if self.hparams.mode != "evaluate":
            self.trainset = Dataset_soundscape(self.hparams, is_train=True).get_ds(mode='train')
            self.valiset = Dataset_soundscape(self.hparams,is_train=False).get_ds(mode='val')
        
        #set path attributes
        self.valid_end_list =[]
        self.rank0_keys = []
        self.rank1_keys = []
        
        #Data modality: Satellite Image # pretrained_model_path,feat_dim=768,fc_dim = 512,global_pool=True, metadata_type='none',meta_droprate=0.5,encoder_type='probSatMAE'
        self.sat_encoder = encoders.sat_encoder(pretrained_model_path=self.hparams.pretrained_model_path,global_pool=True,fc_dim= self.hparams.fc_dim,
                                                metadata_type=self.hparams.metadata_type,meta_droprate=self.hparams.meta_droprate,
                                                encoder_type=self.hparams.sat_encoder_type)
        self.sat_encoder.train()
        #Data modality: Audio 
        if 'audio' in self.hparams.modality_type:
            self.audio_encoder = encoders.audio_encoder(fc_dim= self.hparams.fc_dim,
                                                        encoder_type=self.hparams.audio_encoder_type, freeze = self.hparams.freeze)
            if not self.hparams.freeze:
                self.audio_encoder.train()
        #Data modality: Text 
        if 'text' in self.hparams.modality_type:
            self.text_encoder = encoders.text_encoder(fc_dim= self.hparams.fc_dim,
                                                        encoder_type=self.hparams.text_encoder_type, freeze = self.hparams.freeze)
            if not self.hparams.freeze:
                self.text_encoder.train()
        
        self.criterion = get_criterion(**get_loss_config(name=self.hparams.loss_type,
                                                         pseudo_match_alpha=self.hparams.pseudo_match_alpha,
                                                         vib_beta=self.hparams.vib_beta))
    def get_batch(self,batch):
        out_dict =    {'key':batch[0],
                      'audio_source':batch[1], 'caption_source':batch[2],
                      'audio':{'input_features':batch[3], 'is_longer':batch[4]}, 
                      'text':{'input_ids':batch[5],'attention_mask':batch[6]},
                      'sat_zoom_level':batch[7],'sat':batch[8],
                      'latlong':batch[9], 'time':batch[10], 'month':batch[11],
                      'time_valid':batch[12], 'month_valid':batch[13]} 
        return out_dict
    
    def get_embeds(self,batch):
        embeds = {'sat_embeddings':None, 'audio_embeddings':None, 'text_embeddings':None}
        if self.hparams.metadata_type != 'none':
            embeds['sat_embeddings']  = self.sat_encoder(batch['sat'],sat_type =self.hparams.sat_type, zoom_level = batch['sat_zoom_level'],
                                                        audio_source=batch['audio_source'], caption_source=batch['caption_source'],
                                                        latlong=batch['latlong'], time=batch['time'], month=batch['month'], 
                                                         time_valid=batch['time_valid'], month_valid=batch['month_valid'])
        else:
            embeds['sat_embeddings']  = self.sat_encoder(batch['sat'],sat_type =self.hparams.sat_type, zoom_level = batch['sat_zoom_level'])
        
        batch_audio = {}
        for key in batch['audio'].keys():
            batch_audio[key] = batch['audio'][key]
        embeds['audio_embeddings'] = self.audio_encoder(batch_audio)
        
        if self.hparams.modality_type == 'sat_audio_text':   
            batch_text = {}
            for key in batch['text'].keys():
                batch_text[key] = batch['text'][key]
            embeds['text_embeddings'] = self.text_encoder(batch_text)
        
        return embeds
    
    def get_loss(self,modality1_emb, modality2_emb, matched):

        if not self.hparams.dist_train:
            loss, loss_dict = self.criterion(modality1_emb, modality2_emb, matched=matched)
        else:
            # code for the distributed training
            rank = dist.get_rank()
            
            loss_dict = {}
            modality2_emb_mean_gat = grad_all_gather(modality2_emb['mean'])

            if self.hparams.probabilistic:
                modality2_emb_std_gat = grad_all_gather(modality2_emb['std'])
            else:
                modality2_emb_std_gat = modality2_emb_mean_gat

            if self.hparams.loss_type == 'infonce':
                modality1_emb_mean_gat = grad_all_gather(modality1_emb['mean'])
                
                modality1_emb_all = torch.cat([modality1_emb['mean']] + modality1_emb_mean_gat[:rank] + modality1_emb_mean_gat[rank + 1:])
                modality2_emb_all = torch.cat([modality2_emb['mean']] + modality2_emb_mean_gat[:rank] + modality2_emb_mean_gat[rank + 1:])
                loss_modality1, loss_dict_modality1 = self.criterion(modality1_emb, {'mean': modality2_emb_all}, distributed=True)
                loss_modality2, loss_dict_modality2 = self.criterion(modality2_emb, {'mean': modality1_emb_all}, distributed=True)
                loss = (loss_modality1 + loss_modality2) / 2
                for k, v in loss_dict_modality1.items():
                    loss_dict[k] = (v + loss_dict_modality2[k]) / 2
            else:
                modality1_emb_mean_gat = grad_all_gather(modality1_emb['mean'])
                modality1_emb_std_gat = grad_all_gather(modality1_emb['std'])
                modality1_emb_all = torch.cat([modality1_emb['mean']] + modality1_emb_mean_gat[:rank] + modality1_emb_mean_gat[rank + 1:])
                modality2_emb_all = torch.cat([modality2_emb['mean']] + modality2_emb_mean_gat[:rank] + modality2_emb_mean_gat[rank + 1:])

                modality1_emb_std_all = torch.cat([modality1_emb['std']] + modality1_emb_std_gat[:rank] + modality1_emb_std_gat[rank + 1:])
                modality2_emb_std_all = torch.cat([modality2_emb['std']] + modality2_emb_std_gat[:rank] + modality2_emb_std_gat[rank + 1:])

                extended_matched = torch.cat(
                    [matched] +
                    [self.hparams.label_smoothing * torch.ones(len(modality1_emb['mean']), len(modality1_emb['mean'])).to(matched.device)] * (len(modality1_emb_mean_gat) - 1),
                    dim=1)

                loss_modality1, loss_dict_modality1 = self.criterion(
                                                                    modality1_emb,
                                                                    {'mean': modality2_emb_all, 'std': modality2_emb_std_all},
                                                            matched=extended_matched)
                
                extended_matched = torch.cat(
                    [matched.T] +
                    [self.hparams.label_smoothing * torch.ones(len(modality1_emb['mean']), len(modality1_emb['mean'])).to(matched.device)] * (len(modality1_emb_mean_gat) - 1),
                    dim=1)
                loss_modality2, loss_dict_modality2 = self.criterion(
                                                                    modality2_emb,
                                                                    {'mean': modality1_emb_all, 'std': modality1_emb_std_all},
                                                                    matched=extended_matched)
                loss = (loss_modality1 + loss_modality2) / 2
                for k, v in loss_dict_modality1.items():
                    loss_dict[k] = (v + loss_dict_modality2[k]) / 2
        return loss, loss_dict

    
    def forward(self, batch):
        batch_size = batch['sat'].shape[0]
        embeds = self.get_embeds(batch)  
        matched = torch.eye(batch_size).to(batch['sat'].device)    
        return embeds, matched
    
    def shared_step(self, batch):
        batch = self.get_batch(batch)
        embeds, matched = self(batch)
        audio_embeddings = embeds['audio_embeddings']
        sat_embeddings = embeds['sat_embeddings']
        text_embeddings = embeds['text_embeddings']
        loss_dict = {'loss_ia':None,'loss_it':None,'loss_at':None,'loss':None}
        
        #Calculate losses
        #geoclap loss
        loss_ia, loss_dict_ia = self.get_loss(modality1_emb=sat_embeddings, modality2_emb=audio_embeddings, matched=matched)
        if self.hparams.probabilistic:
            loss_dict['ia_mu_pdist'] = loss_dict_ia['loss/mu_pdist']
            loss_dict['ia_sigma_pdist'] = loss_dict_ia['loss/sigma_pdist']
        loss_dict['loss_ia'] = loss_ia
        if self.hparams.modality_type == 'sat_audio_text':
            loss_it, loss_dict_it = self.get_loss(modality1_emb=sat_embeddings, modality2_emb=text_embeddings, matched=matched)
            loss_at, loss_dict_at = self.get_loss(modality1_emb=audio_embeddings, modality2_emb=text_embeddings, matched=matched)
            
            loss_dict['loss_it'] = loss_it
            loss_dict['loss_at'] = loss_at
            if self.hparams.probabilistic:
                loss_dict['it_mu_pdist'] = loss_dict_it['loss/mu_pdist']
                loss_dict['it_sigma_pdist'] = loss_dict_it['loss/sigma_pdist']
                loss_dict['ia_mu_pdist'] = loss_dict_ia['loss/mu_pdist']
                loss_dict['ia_sigma_pdist'] = loss_dict_ia['loss/sigma_pdist']
                loss_dict['at_mu_pdist'] = loss_dict_at['loss/mu_pdist']
                loss_dict['at_sigma_pdist'] = loss_dict_at['loss/sigma_pdist']
            
            if self.hparams.dataset_type == "soundingEarth":
                loss = (loss_it + loss_at + loss_ia)/3
            else:
                loss_with_text = loss_it + loss_at
                loss = (1-self.hparams.loss_text_weight)*loss_ia + self.hparams.loss_text_weight*loss_with_text
        else:
            loss = loss_ia

        loss_dict['loss'] = loss
        
        out_dict = {'loss_dict':loss_dict, 'embeds':embeds}
        return out_dict
    
    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        if self.hparams.modality_type == 'sat_audio':
            self.log('train_loss', outputs['loss_dict']['loss'].detach(), sync_dist=True, batch_size=self.hparams.train_batch_size)
        if self.hparams.modality_type == 'sat_audio_text':
            self.log('train_loss', outputs['loss_dict']['loss'].detach(), sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('train_loss_ia', outputs['loss_dict']['loss_ia'].detach(), sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('train_loss_it', outputs['loss_dict']['loss_it'].detach(), sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('train_loss_at', outputs['loss_dict']['loss_at'].detach(), sync_dist=True, batch_size=self.hparams.train_batch_size)
        if self.hparams.probabilistic:
            if 'text' in self.hparams.modality_type:
                self.log('it_mu_pdist', outputs['loss_dict']['it_mu_pdist'])
                self.log('it_sigma_pdist', outputs['loss_dict']['it_sigma_pdist'])
                self.log('at_mu_pdist', outputs['loss_dict']['at_mu_pdist'])
                self.log('at_sigma_pdist', outputs['loss_dict']['at_sigma_pdist'])
            self.log('ia_mu_pdist', outputs['loss_dict']['ia_mu_pdist'])
            self.log('ia_sigma_pdist', outputs['loss_dict']['ia_sigma_pdist'])
           
        rank = dist.get_rank()
        if rank == 0:
            self.rank0_keys = self.rank0_keys + batch[0]
        elif rank == 1:
            self.rank1_keys = self.rank1_keys + batch[0]
        
        return outputs['loss_dict']['loss']
        
    
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        val_loss = outputs['loss_dict']
        self.log('val_loss', val_loss['loss'].detach(), sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
        self.log('val_loss_ia', val_loss['loss_ia'].detach(), sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
        if 'text' in self.hparams.modality_type:
            self.log('val_loss_it', val_loss['loss_it'].detach(), sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
            self.log('val_loss_at', val_loss['loss_at'].detach(), sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
     
        self.valid_end_list.append(outputs)
        return outputs

    #compute retrieval metrics for a random batch of validation 
    def on_validation_epoch_end(self):
        print("Duplicate keys between two gpus?",set(self.rank0_keys).intersection(set(self.rank1_keys)))
        outputs = self.valid_end_list
        sat_embeddings_mean = []
        audio_embeddings_mean = []
        sat_embeddings_std = []
        audio_embeddings_std = []

        for i in range(len(outputs)):
            sat_embeddings_mean.append(outputs[i]['embeds']['sat_embeddings']['mean'])
            audio_embeddings_mean.append(outputs[i]['embeds']['audio_embeddings']['mean'])
            if self.hparams.loss_type != 'infonce':
                sat_embeddings_std.append(outputs[i]['embeds']['sat_embeddings']['std'])
                audio_embeddings_std.append(outputs[i]['embeds']['audio_embeddings']['std'])
        sat_embeddings_mean = torch.cat(sat_embeddings_mean,axis=0)
        audio_embeddings_mean = torch.cat(audio_embeddings_mean,axis=0)
        
        if self.hparams.loss_type != 'infonce':
            sat_embeddings_std = torch.cat(sat_embeddings_std,axis=0)
            audio_embeddings_std = torch.cat(audio_embeddings_std,axis=0)
        sat_embeddings_dict = {'mean':sat_embeddings_mean,'std':sat_embeddings_std}
        audio_embeddings_dict = {'mean':audio_embeddings_mean,'std':audio_embeddings_std}

        R_k = self.hparams.recall_at/100*sat_embeddings_mean.shape[0] # Validation with Recall@
        retrieval_results_I2S = get_retrevial_metrics(modality1_emb=sat_embeddings_dict, modality2_emb=audio_embeddings_dict, normalized=False,k=R_k, loss_type=self.hparams.loss_type)
        retrieval_results_S2I = get_retrevial_metrics(modality1_emb=audio_embeddings_dict, modality2_emb=sat_embeddings_dict, normalized=False,k=R_k, loss_type=self.hparams.loss_type)
        self.log(f'I2S_Recall', retrieval_results_I2S['R@'+str(R_k)])
        self.log(f'I2S_Median_Rank', retrieval_results_I2S['Median Rank'])
        
        self.log(f'S2I_Recall', retrieval_results_S2I['R@'+str(R_k)])
        self.log(f'S2I_Median_Rank', retrieval_results_S2I['Median Rank'])
        self.log(f'S2I_Median_Rank', retrieval_results_S2I['Median Rank'])
        # import code; code.interact(local=dict(globals(), **locals()))
        self.valid_end_list = []
        self.rank0_keys = []
        self.rank1_keys = []
        return retrieval_results_I2S, retrieval_results_S2I
       

    def train_dataloader(self):
        trainloader = wds.WebLoader(self.trainset, batch_size=None,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers)
        return trainloader

    def val_dataloader(self):
        valloader = wds.WebLoader(self.valiset, batch_size=None, #batch_size=self.hparams.val_batch_size,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers)
        return valloader

    # def test_dataloader(self):
    #     testloader = wds.WebLoader(self.testset, batch_size=None, #batch_size=self.hparams.val_batch_size,
    #                 shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers)
    #     return testloader

    # def on_before_optimizer_step(self, optimizer):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')
        
        params = self.parameters()
        
        self.optim = torch.optim.AdamW(params=params,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9,0.98),
                    eps=1e-6
                    )
         
        self.warm_up_iterations = self.hparams.warm_up_iterations
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim,
            T_0 = self.warm_up_iterations
        )
        return {'optimizer': self.optim,
        'lr_scheduler': {
            'name':'train/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }