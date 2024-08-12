import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tqdm import tqdm
from ftlangdetect import detect


parser = argparse.ArgumentParser()
parser.add_argument('--llm', type=str, choices=['gpt', 'llama'],default='gpt')
parser.add_argument('--dataset_type', type=str, choices=['geoclap', 'sat2sound'],default='geoclap')
parser.add_argument('--df_path', type=str, default="")
parser.add_argument('--save_path', type=str, default="")
parser.add_argument('--stride', type=int, default=16) #context length/ size of sliding window
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
print(args.dataset_type)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = None
tokenizer = None

if args.llm == 'gpt':
    model_id = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
elif args.llm == 'llama':
    access_token = "hf_qiHjtESTKVyNOeWMbwNxgfxQnifNLBMXap"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)
else:
    raise Exception('Model Not Found')

model.to(device)

df = pd.read_csv(args.df_path)
if args.dataset_type == "geoclap":
    texts = list(df['description'].fillna("This is the sound"))
else:
    texts = list(df['text'].fillna("This is the sound"))

ppl_score = []
lang = []
lang_score = []
for text in tqdm(texts):
    lang_result = detect(text=text.replace("\n"," "), low_memory=True)
    lang.append(lang_result['lang'])
    lang_score.append(lang_result['score'])
    encodings = tokenizer("\n\n".join(text), return_tensors="pt")

    max_length = 4096
    if args.llm != 'llama':
        max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, args.stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    ppl_score.append(ppl)

    
df['ppl_score'] = ppl_score
df['lang_score'] = lang_score
df['lang'] = lang

df.to_csv(args.save_path)