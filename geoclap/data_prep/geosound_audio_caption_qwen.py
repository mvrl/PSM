# git clone https://github.com/QwenLM/Qwen-Audio.git
# Follow Qwen-Audio's README.md to create necessary environment.
# copy this script to: /Qwen-Audio/
import pandas as pd
import torch
from tqdm import tqdm
import os
import json


### CAPTIONER SPECIFIC CODE ####
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(1234)

MODEL = "qwen"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", device_map="cuda:0", trust_remote_code=True).eval()
sp_prompt = "<|startofanalysis|><|en|><|caption|><|en|><|notimestamps|><|wo_itn|>"

def get_caption(audio_path):
    try:
        query = f"<audio>{audio_path}</audio>{sp_prompt}"
        audio_info = tokenizer.process_audio(query)
        inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info)
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs, audio_info=audio_info)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False,audio_info=audio_info).split("<|wo_itn|>")[1].split("<|endoftext|>")[0]
    except:
        response = ""
    return response

### CAPTIONER SPECIFIC CODE ####

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

if __name__ == '__main__':
    output_json_file = "./geosound_audio_caption_"+MODEL+".json"
    data_path = "/storage1/jacobsn/Active/user_k.subash/data_raw/"
    train_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/train_metadata.csv")
    val_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/val_metadata.csv")
    test_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/test_metadata.csv")
    aporee_meta = pd.read_csv("/storage1/jacobsn/Active/user_k.subash/data_raw/aporee/final_metadata_with_captions.csv")

    df_final = pd.concat([train_df,val_df, test_df])
    for i in tqdm(range(len(df_final))):
        sample = df_final.iloc[i]
        sample_id = sample['sample_id']
        source = sample_id.split("-")[0]
        key = sample_id.split("-")[1]
        sound_format =  'mp3'
        if source == 'aporee':   
            soundname = aporee_meta[aporee_meta['long_key']==key].mp3name.item()
            audio_path = os.path.join(data_path,source,'raw_audio',str(key),soundname)
        else:
            if isinstance(key, str):
                soundname = key+"."+sound_format
            else:
                soundname = str(key)+"."+sound_format
            
            audio_path = os.path.join(data_path,source,'raw_audio',soundname)
        
        caption = get_caption(audio_path)
        if len(caption) == 0: #if the captioner returns nothing
            caption = "This sound is a sound of something."
        output_dict = {'sample_id':sample_id,MODEL+"_caption":caption}
        save_dict_to_json(dictionary=output_dict, output_file=output_json_file)