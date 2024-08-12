# For audios from iNat, we have different sound format. This script converts them all into .mp3 format. This makes it easier while creating webdataset.
import os
import pandas as pd
from collections import Counter
from pydub import AudioSegment
import shutil
from tqdm import tqdm

data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data_raw"
inat_original_audios = os.path.join(data_path,"iNat/raw_audio_original")
inat_output_audios = os.path.join(data_path,"iNat/raw_audio")

train_df = pd.read_csv(os.path.join(data_path,"train_metadata.csv"))
val_df = pd.read_csv(os.path.join(data_path,"val_metadata.csv"))
test_df = pd.read_csv(os.path.join(data_path,"test_metadata.csv"))

meta_df = pd.concat([train_df, val_df, test_df])
meta_inat = meta_df[(meta_df['source']=='iNat')]
print(Counter(meta_inat['sound_format'])) #Counter({'m4a': 51155, 'wav': 40628, 'mp3': 21838, 'mpga': 983})

for i in tqdm(range(len(meta_inat))):
    try:
        sample = meta_inat.iloc[i]
        audioname = str(sample.key)+"."+sample.sound_format
        audio_path = os.path.join(inat_original_audios,audioname)
        if sample.sound_format == 'wav' or sample.sound_format == 'm4a':
            sound = AudioSegment.from_file(audio_path, format=sample.sound_format)
            sound.export(os.path.join(inat_output_audios,str(sample.key)+".mp3"), format="mp3")
        elif sample.sound_format == 'mpga' or sample.sound_format == 'mp3':
            dst_audio_path = os.path.join(inat_output_audios,str(sample.key)+".mp3")
            shutil.copy(audio_path, dst_audio_path)
    except:
        print("Failed for:", audioname)