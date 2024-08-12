import pandas as pd
from tqdm import tqdm
import os
import json
import code
import torch
from transformers import ClapAudioModelWithProjection, ClapProcessor, AutoProcessor, ClapModel
from transformers import AutoTokenizer, ClapTextModelWithProjection
import torchaudio
import torch.nn.functional as F
import numpy as np

model = ClapModel.from_pretrained("laion/clap-htsat-fused").to("cpu")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
logit_scale_a = np.exp(model.logit_scale_a.detach().cpu().numpy())
logit_scale_t = np.exp(model.logit_scale_t.detach().cpu().numpy())

SAMPLE_RATE = 48000

def get_processed_audio(audio_path):
    track, sr = torchaudio.load(audio_path)
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    return track

def get_score(audio,caption):
    caption = ' '.join(caption.split()[:77])
    try:
        inputs = processor(text=caption, audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs['is_longer'] = torch.tensor([True])
        for k in inputs.keys():
            inputs[k] = inputs[k].to(model.device)
        outputs = model(**inputs)
        sim_a2t = outputs.logits_per_audio[0][0].detach() / logit_scale_a
        sim_t2a = outputs.logits_per_text[0][0].detach() / logit_scale_t
        return sim_a2t.detach().cpu().numpy().item(), sim_t2a.detach().cpu().numpy().item()
    except:
        inputs = processor(text="This is a sound", audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs['is_longer'] = torch.tensor([True])
        for k in inputs.keys():
            inputs[k] = inputs[k].to(model.device)
        outputs = model(**inputs)
        sim_a2t = outputs.logits_per_audio[0][0].detach() / logit_scale_a
        sim_t2a = outputs.logits_per_text[0][0].detach() / logit_scale_t
        return sim_a2t.detach().cpu().numpy().item(), sim_t2a.detach().cpu().numpy().item()

def process_sample(sample):
    output_dict = {'sample_id': None, 
                'clap_a2t_meta':None, 'clap_t2a_meta':None,
                'clap_a2t_pengi':None, 'clap_t2a_pengi':None,
                'clap_a2t_qwen':None, 'clap_t2a_qwen':None}

    sample_id = sample['sample_id']
    source = sample_id.split("-")[0]
    key = sample_id.split("-")[1]
    meta_caption = sample['text']
    pengi_caption = pengi_df[pengi_df['sample_id'] == sample_id]['pengi_caption'].item()
    qwen_caption = qwen_df[qwen_df['sample_id'] == sample_id]['qwen_caption'].item()
    if type(meta_caption) != str or len(meta_caption) == 0:
        meta_caption = "This is a sound."

    sound_format = 'mp3'
    if source == 'aporee':
        soundname = aporee_meta[aporee_meta['long_key'] == key].mp3name.item()
        audio_path = os.path.join(data_path, source, 'raw_audio', str(key), soundname)
    else:
        if isinstance(key, str):
            soundname = key + "." + sound_format
        else:
            soundname = str(key) + "." + sound_format

        audio_path = os.path.join(data_path, source, 'raw_audio', soundname)

    audio = get_processed_audio(audio_path)

    output_dict['sample_id'] = sample_id
    # for metadata caption
    output_dict['clap_a2t_meta'], output_dict['clap_t2a_meta'] = get_score(audio, meta_caption)

    # for pengi caption
    output_dict['clap_a2t_pengi'], output_dict['clap_t2a_pengi'] = get_score(audio, pengi_caption)
    
    # for qwen caption
    output_dict['clap_a2t_qwen'], output_dict['clap_t2a_qwen'] = get_score(audio, qwen_caption)
   
    return output_dict

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')

if __name__ == '__main__':
    output_json_file = "/scratch/k.subash/projects/clap_score_geosound.json"
    data_path = "/storage1/jacobsn/Active/user_k.subash/data_raw/"
    train_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/train_metadata.csv")
    val_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/val_metadata.csv")
    test_df = pd.read_csv("/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/test_metadata.csv")
    aporee_meta = pd.read_csv(
        "/storage1/jacobsn/Active/user_k.subash/data_raw/aporee/final_metadata_with_captions.csv")
    pengi_df = pd.read_json(
        "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/geosound_audio_caption_pengi.json",
        lines=True)
    qwen_df = pd.read_json(
        "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/GeoSound/geosound_audio_caption_qwen.json",
        lines=True)

    df_final = pd.concat([train_df, val_df, test_df])
    
    for i in tqdm(range(len(df_final))):
        sample = df_final.iloc[i]
        result = process_sample(sample)
        save_dict_to_json(dictionary=result, output_file=output_json_file)