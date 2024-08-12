from transformers import ClapProcessor
import torchaudio
import torch
import os
from transformers import RobertaTokenizer
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

tokenize = RobertaTokenizer.from_pretrained('roberta-base')
def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}

def get_text_clap(text):#accepts list of text prompts
    text = tokenizer(text)
    return text


processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
SAMPLE_RATE = 48000

def get_audio_clap(track,sr,padding="repeatpad",truncation="fusion"):
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    output = processor(audios=track, sampling_rate=SAMPLE_RATE, max_length_s=10, return_tensors="pt",padding=padding,truncation=truncation)
    return output


if __name__ == '__main__':
    import random
    path_to_audio = '/storage1/fs1/jacobsn/Active/user_k.subash/data_archive/iNat/raw_audio'
    files = os.listdir(path_to_audio)
    path_to_audio = os.path.join(path_to_audio,files[random.randint(1, len(files))])
    track, sr = torchaudio.load(path_to_audio)  # Faster!!!
    samples = []
    tick = time.time()
    for i in range(5):
        sample =  get_audio_clap(track,sr)
        samples.append(sample)
    # print(sample.keys())
    # print(sample['input_features'].shape,sample['is_longer'].shape)
    # print(sample['is_longer'])
    # print(get_text_clap(['dummy text'])['input_ids'].shape)
    # print(get_text_clap(['dummy text'])['attention_mask'].shape)
    tock = time.time()
    time_taken = tock - tick
    print(f'The total time taken is {time_taken}')
    import code;code.interact(local=dict(globals(), **locals()))
    
    



