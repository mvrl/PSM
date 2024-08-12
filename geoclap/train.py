import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import torch
import os
import random
from argparse import ArgumentParser
import sys
import warnings
from .engine import GeoCLAPModel
from .config import cfg

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "300"

def get_shards(dataset_type="GeoSound",overhead_type="sentinel"):
    if dataset_type == "GeoSound":
        data_path = os.path.join(cfg.GeoSound_webdataset_path,"with_"+overhead_type)
    else:
        data_path = cfg.SoundingEarth_webdataset_path
    all_shards = [os.path.join(data_path,s) for s in os.listdir(data_path) if ".tar" in s]
    test_shard = [s for s in all_shards if 'test' in s]
    val_shard = [s for s in all_shards if 'val' in s]
    train_shards = [s for s in all_shards if 'train' in s]
    return train_shards, val_shard, test_shard

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


def get_args():
    parser = ArgumentParser(description='')
    #training hparams
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--train_epoch_length', type=int, default=10000)
    parser.add_argument('--limit_val_batches', type=int, default=30)
    parser.add_argument('--val_check_interval', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)

    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--mode', type=str, default='dev', choices=['dev', 'train'])   
    
    parser.add_argument('--dataset_type', type=str, default='GeoSound',choices=['GeoSound','SoundingEarth'])
    parser.add_argument('--modality_type', type=str, default='sat_audio_text')
    parser.add_argument('--caption_strategy', type=str, default='original',choices=['original','meta','pengi','qwen'])
    parser.add_argument('--sat_input_size', type=int, default= 224)
    parser.add_argument('--sat_type', type=str, default='sentinel', choices=['sentinel','bingmap','googleEarth']) 
    parser.add_argument('--metadata_type', type=str, default='latlong_month_time_asource_tsource',help="'latlong', 'month', 'latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource', 'none'")
    parser.add_argument('--meta_droprate', type=float, default=0.5)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warm_up_iterations', type=int, default=5000)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
   
    parser.add_argument('--accelerator',type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    
    parser.add_argument('--project_name', type=str, default='improvedPSM')
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    
    # encoder types:
    parser.add_argument('--sat_encoder_type',type=str,default='probSatMAE', choices=['baselineSatMAE, probSatMAE']) 
    parser.add_argument('--audio_encoder_type',type=str,default='probCLAP', choices=['baselineCLAP', 'probCLAP'])
    parser.add_argument('--text_encoder_type',type=str,default='probCLAP',choices=['baselineCLAP', 'probCLAP'])
    parser.add_argument('--freeze',type=str,default='false',choices=['true', 'false']) #freeze CLAP or not.
    
    parser.add_argument('--fc_dim', type=int, default = 512)

    parser.add_argument('--probabilistic',type=str,default='true',choices=['true', 'false'])
    parser.add_argument('--label_smoothing', type=float, default=0.0001)
    parser.add_argument('--pseudo_match_alpha', type=float, default=0.1)
    parser.add_argument('--vib_beta', type=float, default=0.0001)

    parser.add_argument('--loss_type',type=str,default='pcmepp',choices=['infonce','pcmepp'])
    parser.add_argument('--loss_text_weight', type=float, default=0.5)
    parser.add_argument('--recall_at', type=int, default = 10) #percent
    # Training resuming parameters:
    parser.add_argument('--ckpt_path',type=str, default ='none')
    parser.add_argument('--ckpt_mode',type=str, default ='hard')

    args = parser.parse_args()

    return args

def set_attrs(args):

    if args.devices > 1:
        setattr(args,'dist_train',True)
    else:
        setattr(args,'dist_train',False)
    
    if args.probabilistic == 'true':
        setattr(args,'probabilistic',True)
        setattr(args,'sat_encoder_type',"probSatMAE")
        setattr(args,'audio_encoder_type',"probCLAP")
        setattr(args,'text_encoder_type',"probCLAP")
        setattr(args,'loss_type',"pcmepp")
    else:
        setattr(args,'probabilistic',False)
        setattr(args,'sat_encoder_type',"baselineSatMAE")
        setattr(args,'audio_encoder_type',"baselineCLAP")
        setattr(args,'text_encoder_type',"baselineCLAP")
        setattr(args,'loss_type',"infonce")
    
    if args.dataset_type == "SoundingEarth":
        setattr(args,'sat_type',"googleEarth")

    if args.freeze == "false":
        setattr(args,'freeze',False)
    else:
        setattr(args,'freeze',True)
    setattr(args,'pretrained_model_path',cfg.satmae_pretrained_ckpt)
    
    return args


if __name__ == '__main__':
    set_seed(56)
    args = get_args()
    args = set_attrs(args)
    train_shards, val_shard, _ = get_shards(overhead_type=args.sat_type, dataset_type=args.dataset_type)
    args.train_path = train_shards
    args.vali_path = val_shard
    #set learning rate logger
    print('Starting Training')
    print(args)
    #initliaze model
    geoclap_model = GeoCLAPModel(args)
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=cfg.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    ckpt_monitor1 = ((
            ModelCheckpoint(monitor='val_loss', mode='min', filename='{epoch}-{step}-{val_loss:.3f}',save_top_k = 10, save_last=True,save_on_train_epoch_end=False)
        ))
    ckpt_monitor2 = ((
            ModelCheckpoint(monitor='I2S_Recall', mode='max',filename='{epoch}-{I2S_Recall:.3f}',save_top_k = 10, save_last=True,save_on_train_epoch_end=False)
        ))
    

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(profiler="simple",precision=16,fast_dev_run=6, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=4,
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitor1, ckpt_monitor2, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision=16, max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitor1, ckpt_monitor2, lr_logger], 
        val_check_interval=args.val_check_interval, check_val_every_n_epoch=None, limit_val_batches=args.limit_val_batches,
        log_every_n_steps=15)
    else:
        raise ValueError('Invalid value for mode')
    
    if args.ckpt_path.lower()=='none'.lower():
        trainer.fit(geoclap_model)
    else:
        if args.ckpt_mode.lower()=='hard':
            print('Hard Checkpoint Reload')
            trainer.fit(geoclap_model, ckpt_path=args.ckpt_path)
        elif args.ckpt_mode.lower()=='soft':
            print('Soft Checkpoint Reload')
            checkpoint = torch.load(args.ckpt_path)
            geoclap_model.load_state_dict(checkpoint['state_dict'])
            trainer.fit(geoclap_model)