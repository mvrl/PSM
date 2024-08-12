# Data Preparation
## Geotagged Audios
We downloaded geotagged sounds from four sources: `aporee`, `freesound`, `iNaturalist`, and `yahoo100m`. 

### 1. yahoo100m
[Yahoo100m](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) offers large multimedia dataset which includes almost $0.8$ million videos. Among these videos around $100k$ videos are geotagged. We will use these geotagged videos as one of the sources of our dataset.

1. Refer to this nice repo: [yfcc100m](https://pypi.org/project/yfcc100m/), and follow the steps to download yfcc100m data. As mentioned in it's `README.md`, it is strongly adviced to setup free [aws account](https://aws.amazon.com/de/blogs/security/wheres-my-secret-access-key/) before proceeding with the download. Downloading data is as easy as following three steps:
    ````
    1. pip install yfc100m

    2. python -m yfcc100m.convert_metadata <input dir> -o <meta dir>

    3. python -m yfcc100m.download <meta dir> -o <zip dir>

    ````
    Note: By default the `yfcc100m.download` downloads images from the metadata files. I changed the default value of argument `--kind` in [L213](https://gitlab.com/jfolz/yfcc100m/-/blob/master/yfcc100m/download.py?ref_type=heads#L213) to `(1,)` and downloaded only videos.

2. We can further extract metadata of the video data in `yfcc100m` dataset into `yfcc100m_metadata.csv` using the following script:
    ```
    python yfcc_metadata_extraction.py
    ```
3. Extract audio from videos of yfcc database using:
    ```
    python yfcc_audio_extraction.py 
    ```
    This script also saves the status of the audio extraction job into a file: `yfcc100m_status.csv`. This will allow us later to filter out the samples for which audio extraction failed for whatever reasons.
### 2. AporeeRadio
Taken from [GeoCLAP](https://github.com/mvrl/Soundscape/tree/geoclap):

we use the audio data collected by the paper: [Self-supervised Audiovisual Representation Learning for Remote Sensing Data](https://arxiv.org/abs/2108.00688) can be downloaded from https://zenodo.org/record/5600379. The downloded data will have high-resolution GoogleEarth imagery along with a `metadata.csv` containing details of the audio recording corresponding to the overhead imagery. 
    
1. Now to download raw audio data use run:\
    ```./get_SoundingEarth_raw_audio.sh```

    Try to make sure that all of the audio files are downloaded (for some files download fails and one might have to repeat download for the remaining audio files).

2. Once all data is downloaded a quick sanity check can be run using the python script `./SoundingEarth_sanity.py` which will just try to read the audio files and save those as torch tensors `.pt`. Moreover, it also saves the id of the audio samples that failed to be read into a file: `corrupt_ids_final.csv`.

3. `./clean_SoundingEarth_metadata.py`: This python script performs simple pre-processing of the `description` column of `metadata.csv`. Moreover, it uses `geopy` to perform reverse geo-coding of the address from the given lattitude-longitude of the audio sample and adds that address along with the cleaned description of the audio. Finally, this script yields a file: `final_metadata_with_captions.csv` containing pre-processed captions along with all other metadata for audio samples in our data. 

### 3. Freesound

[www.freesound.org](https://freesound.org) hosts large number of sounds, out of which as we collected around $50k$ geotagged audios. To download sounds one must first apply for [API](https://freesound.org/apiv2/apply).

1. To get a list of geotagged freesound IDs saved into a file: `freesound_geotagged_IDs.csv`, run:
    ```
    python geotagged_freesound_ids.py
    ```
    
2. Now download those audio files using:
    ```
    python freesound_download.py --api_key YOUR_API_KEY --csv_file path_to_freesound_geotagged_IDs.csv --split_id $n$ --output_directory path_to_save_downloads
    ```

### 4. iNaturalist
[www.inaturalist.org](https://www.inaturalist.org) hosts large collection of geotagged observations especially focused on biodiversity around the world. We select observations with following filters on: `Verifiable`, `Research Grade`, and `Has Sounds`. We download multiple split `.csv` files (each not exceeding $200$ k rows) and combined them to get a `iNaT_metadata.csv` containing over $450$ k rows. 

1. We use the metadata file to download iNaturalist sounds using:

    ```
    python iNaturalist_download.py --csv_file iNaT_metadata.csv --split_id n --output_directory path_to_download
    ``` 
2. In order to create a source balanced dataset, we decided to take only $120$ k samples from `iNaturalist`. Moreover, to get relatively species-balanced samples, we run the following script:

    ```
    python iNaturalist_sampling.py 
    ```

    This will create a csv `iNaT_metadata_species_balanced.csv`.

### Metadata Merging & train/val/test split
Finally, we merge the geolocations of the sounds from all four sources. 

1. We merge the geolocations of all of our data samples using:
    ```
    python merge_geolocations.py
    ```
    This leads to the data distribution of:\
    {'iNat': 120016, 'yfcc': 98506, 'aporee': 50792, 'freesound': 49677} #total count: 318991
    with geolocation, source, and fileID saved in a file `source_balanced_geotagged_sounds.csv`

    Overhead imagery from two sources: `bingmap` and `sentinel2-cloudless` were downloaded for these samples.

2. Finally, to get rid of potentially corrupt files, a simple file-size based sanity check is done on satellite imagery using:
    ```
    python images_sanity.py
    ```
    This leads to the data distribution of:\
    {'iNat': 119899, 'yfcc': 98236, 'aporee': 50784, 'freesound': 49627} #total count: 318546
    with geolocation, source, and fileID saved in a file `source_balanced_filtered_geotagged_sounds.csv`

3. We can also get the physical address for most of the geolocations using reverse geocoding as used in the script:

    ```
    python get_address.py
    ```

    This adds an extra column `address` as reflected in a csv `source_balanced_filtered_geotagged_sounds_address.csv`

2. Some basic pre-processing of textual metadata from all four sources is done using:
    ```
    python clean_text.py
    ```

3. Merge metadata for data from all four sources using:

    ```
    python merge_metadata.py
    ```
    This merges the key metadata for a total of 318,546 samples into a common file `merged_metadata_final.csv` with following source distribution: {'iNat': 119899, 'yfcc': 98236, 'aporee': 50784, 'freesound': 49627}

4. Final sanity check of overall dataset is done using:

    ```
    python data_sanity.py
    ```
    This simply reads audio and image for all of the observations in `merged_metadata_final.csv` and saves status of each read into `dataset_sanity.csv`.

5. Using `merged_metadata_final.csv` and `dataset_sanity.csv`, the overall dataset is split into train/val/test split using: 
    ```
    python data_split.py
    ```

    Using our designed data split strategy, this script splits data into 294113/5000/10000 samples for train/val/test. 

    Train split source distribution: `{'iNat': 108753, 'yfcc': 92055, 'aporee': 46893, 'freesound': 46318}`\
    Val split source distribution: `{'iNat': 1851, 'yfcc': 1565, 'aporee': 797, 'freesound': 787}`\
    Test split source distribution: `{'iNat': 3999, 'yfcc': 2832, 'aporee': 1594, 'freesound': 1575}`

    TRAIN/VAL/TEST : 294019/5000/10000

    Metadata for these splits is saved in files: `train_metadata.csv`, `val_metadata.csv`, and `test_metadata.csv`
6. Audio samples from `iNaturalist` have different file formats. All those audios are converted to `.mp3` format using:
    ```
    python convert_audio.py
    ```

## Overhead Imagery

We download overhead imagery from two sources `Sentinel2-cloudless` and `BingMaps`.
### 1. BingMaps

1. For `BingMaps`, once the free student [api-key](https://www.microsoft.com/en-us/maps/create-a-bing-maps-key) is obtained, we can download overhead imagery for all geolocations listed in the file `source_balanced_geotagged_sounds.csv` by using:

    ```
    python download_bing.py --api_key YOUR_BINGMAP_API_KEY \
                            --merged_latlong_csv path_to_source_balanced_geotagged_sounds.csv \
                            --out_dir path_to_download \
                            --split_id $n$ 
    ```
    Note: Pay attention to the download LIMIT per day (set as a GLOBAL variable in the script) to avoid possible ban!

### 2. Sentinel

We use `mapproxy` to download sentinel imagery for all the geolocations of our interest. Follow the [README.md](https://github.com/mvrl/sat2sound/blob/main/geoclap/data_prep/CVGlobal/README.md) of `./CVGlobal` to download sentinel2-cloudless imagery.

## Webdataset Creation

For the ease of sharing and faster training. We finally create `webdataset` `.tar` files for `train/val/test` split of our data using:

```
python create_webdataset --overhead sentinel --split train
python create_webdataset --overhead sentinel --split val
python create_webdataset --overhead sentinel --split test

python create_webdataset --overhead bingmap --split train
python create_webdataset --overhead bingmap --split val
python create_webdataset --overhead bingmap --split test

```
