# git clone https://github.com/microsoft/Pengi.git
# Follow PENGI's README.md to create necessary environment.
# copy this script to: /Pengi/

import pandas as pd
from tqdm import tqdm
import os
import json


### CAPTIONER SPECIFIC CODE ####
from wrapper import PengiWrapper as Pengi

MODEL = "pengi"

pengi = Pengi(config="base") #base or base_no_text_enc

def get_caption(audio_path):
    try:
        audio_file_paths = [audio_path]
        text_prompts = ["generate metadata"]
        add_texts = [""]
        generated_response = pengi.generate(
                                            audio_paths=audio_file_paths,
                                            text_prompts=text_prompts, 
                                            add_texts=add_texts, 
                                            max_len=77, 
                                            beam_size=1, 
                                            temperature=1.0, 
                                            stop_token=' <|endoftext|>',
                                            )
        response = generated_response[0][0][0]
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