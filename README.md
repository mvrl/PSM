Implementation of **"PSM: Learning Probabilistic Embeddings for Multi-scale Zero-shot Soundscape Mapping"** , accepted at ACM MM 2024.\
[arxiv](https://arxiv.org/abs/2309.10667)

[data](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link)

[model-checkpoints](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link)


1. Clone this repo
    ```
    git clone git@github.com:mvrl/PSM.git
    cd PSM/geoclap
    ```
2. Setting up enviornment
    ```
    conda env create --file environment.yml
    conda activate sat2audio
    ```
    
    Note: Instead of `conda` it could be easier to pull docker image `ksubash/sat2audio:2.0` for the project we provide using following steps:

    ```
    docker pull ksubash/sat2audio:2.0
    docker run -v $HOME:$HOME --gpus all --shm-size=64gb -it ksubash/geoclap
    source /opt/conda/bin/activate /opt/conda/envs/sat2audio_demo
    ```

3. Please refer to `./data_prep/README.md` for details on SoundingEarth and instructions on how to download Sentinel2 imagery. Some scipts for basic pre-processing steps required for experiments related to `PSM` are also provided there.

4. Copy the pre-trained checkpoint of `SATMAE` named as `pretrain-vit-base-e199.pth` provided in [this google drive folder](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link) to the location pointed by `cfg.satmae_pretrained_ckpt`.

5. Check both `config.py` and `./data_prep/config.py` to setup relevant paths by manually creating relevant directories. 

5. Now assuming that the data is downloaded and paths in `config.py` are properly setup, we are now ready to run experiments related to GeoCLAP. Change directory by one step in hierarchy so that `geoclap` can be run as a python module.
    ```
    cd ../
    ```

6. Assuming [wandb](https://wandb.ai/home) is set up correctly for logging purpose, we can launch the PSM training as follows:
    ```
   python -m geoclap.train --num_workers 8 \
                            --probabilistic true \
                            --metadata_type \
                            --latlong_month_time_asource_tsource \
                            --run_name GeoSound_pcmepp_metadata_sentinel \
                            --dataset_type GeoSound \
                            --sat_type sentinel \
                            --mode train \
                            --wandb_mode online
    
7. Once the training is complete and we have the appropriate checkpoint of the model, we can evaluate the cross-modal retrevial performance of the model. For example,
    ```
    python -m geoclap.evaluate --ckpt_path GeoSound_pcmepp_metadata_sentinel_best_ckpt_path \
                               --loss_type pcmepp \
                               --dataset_type GeoSound \
                               --test_zoom_level 0 \
                               --sat_type sentinel \
                               --metadata_type latlong_month_time_asource_tsource \
                               --add_text true \
                               --meta_droprate 0 \
                               --test_mel_index 0 
    ```

**Citation:**
```
@inproceedings{khanal2024psm,
  title = {PSM: Learning Probabilistic Embeddings for Multi-scale Zero-shot Soundscape Mapping},
  author = {Khanal, Subash and Eric, Xing and Sastry, Srikumar and Dhakal, Aayush and Xiong Zhexiao and Ahmad, Adeel and Jacobs, Nathan},
  year = {2024},
  month = nov,
  booktitle = {Association for Computing Machinery Multimedia (ACMMM)},
}
```

Follow more works from our lab here: [The Multimodal Vision Research Laboratory (MVRL)](https://mvrl.cse.wustl.edu)
