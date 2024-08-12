import os
ckpt_cfg = {}

wandb_log_path = "./logs/best_ckpts"
#ckpts for different experiments:
ckpt_cfg["GeoSound_infonce_bingmap"] = os.path.join(wandb_log_path,"zgameb0q","epoch=1-I2S_Recall=0.436.ckpt") 
ckpt_cfg["GeoSound_infonce_metadata_bingmap"] = os.path.join(wandb_log_path,"n5v3n2a4","epoch=1-I2S_Recall=0.564.ckpt") 
ckpt_cfg["GeoSound_pcmepp_bingmap"] = os.path.join(wandb_log_path,"n000csi9","epoch=1-I2S_Recall=0.468.ckpt") 
ckpt_cfg["GeoSound_pcmepp_metadata_bingmap"] = os.path.join(wandb_log_path,"15q3k37e","epoch=0-I2S_Recall=0.666.ckpt") 

ckpt_cfg["GeoSound_infonce_sentinel"] = os.path.join(wandb_log_path,"ru3pbrz1","epoch=1-I2S_Recall=0.465-v1.ckpt")  
ckpt_cfg["GeoSound_infonce_metadata_sentinel"] = os.path.join(wandb_log_path,"kjnvgxhr","epoch=0-I2S_Recall=0.568.ckpt") 
ckpt_cfg["GeoSound_pcmepp_sentinel"] = os.path.join(wandb_log_path,"k81mf7ro","epoch=0-I2S_Recall=0.497.ckpt") 
ckpt_cfg["GeoSound_pcmepp_metadata_sentinel"] = os.path.join(wandb_log_path,"z2tpbd70","epoch=0-I2S_Recall=0.666.ckpt") 


ckpt_cfg["SoundingEarth_infonce_googleEarth"] = os.path.join(wandb_log_path,"x8iwy4a2","epoch=0-I2S_Recall=0.434.ckpt") 
ckpt_cfg["SoundingEarth_infonce_metadata_googleEarth"] = os.path.join(wandb_log_path,"8lzdvlan","epoch=0-I2S_Recall=0.451.ckpt") 
ckpt_cfg["SoundingEarth_pcmepp_googleEarth"] = os.path.join(wandb_log_path,"mfx8omr1","epoch=0-I2S_Recall=0.469.ckpt") 
ckpt_cfg["SoundingEarth_pcmepp_metadata_googleEarth"] = os.path.join(wandb_log_path,"1dh3g756","epoch=0-I2S_Recall=0.502.ckpt") 


ckpt_cfg["results_json"] = "./logs/results/PSM_RESULTS.json"