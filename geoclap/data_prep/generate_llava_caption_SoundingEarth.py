#Using LLAVA to generate sound related captions for the SoundingEarth dataset.
import requests
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
import time


def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('--overhead', type=str, default="googleEarth",choices=["sentinel","bingmap","googleEarth"])
    args = parser.parse_args()

    data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/aporee"
    train_df = pd.read_csv(os.path.join(data_path,"aporee_train_fairsplit.csv"))
    val_df =  pd.read_csv(os.path.join(data_path,"aporee_val_fairsplit.csv"))
    test_df = pd.read_csv(os.path.join(data_path,"aporee_test_fairsplit.csv"))

    df_final = pd.concat([train_df,val_df, test_df])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("running device:",device)
    model_id = "llava-hf/llava-1.5-7b-hf"
    prompt_text = "What types of sounds can we expect to hear from the location captured by this aerial view image? Describe in up to two sentences."
    prompt = "USER: <image>\n"+prompt_text+"\nASSISTANT:"
    model = LlavaForConditionalGeneration.from_pretrained(
                                                          model_id, 
                                                          torch_dtype=torch.float16, 
                                                          low_cpu_mem_usage=False, 
                                                          ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def get_caption(image_path):
      image = Image.open(image_path)
      try:
        inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=77, do_sample=False)
        caption = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[1]
      except:
        caption = "This is a sound of some place."
      return caption

    output_json_file = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw/metafiles/SoundingEarth/SoundingEarth_llava_caption_for_"+str(args.overhead)+".json"
    
    for i in tqdm(range(len(df_final))):
      sample = df_final.iloc[i]
      mp3name = sample['mp3name']
      short_id = sample['key']
      long_id = sample['long_key']
      
      if args.overhead == 'googleEarth':
          image_path = os.path.join(data_path,'images','googleEarth',str(short_id)+'.jpg')
      elif args.overhead == 'sentinel':
          image_path = os.path.join(data_path,'images','sentinel_geoclap',str(short_id)+'.jpeg')
      elif args.overhead == 'bingmap':
          image_path = os.path.join(data_path,'images','bingmap_geoclap',str(long_id)+'.jpg')
      else:
          raise NotImplementedError("supported satellite image types are:[googleEarth, sentinel, bingmap]")
      
      captions = get_caption(image_path)
      output_dict = {'sample_id':long_id,"captions":captions}
      save_dict_to_json(dictionary=output_dict, output_file=output_json_file)
      

