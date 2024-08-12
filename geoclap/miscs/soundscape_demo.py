import os
from ..demo_config import demo_cfg


def run_region_image_download(log_dir, region_name, latitude, longitude, spacing, max_distance):
    # Run region_image_download.py
    os.system(f"python -m geoclap.miscs.region_image_download \
        --log_dir {log_dir} \
        --region_name {region_name} \
        --latitude {latitude} \
        --longitude {longitude} \
        --spacing {spacing} \
        --max_distance {max_distance}")

def run_soundscape_mapping_small(output_dir, region_name, sat_type, month, hour, zoom_level,
                                 asource, tsource, metadata_type, query_type, text_query_file,
                                 audio_query_file, expr):
    # Run soundscape_mapping_small.py
    os.system(f"python -m geoclap.miscs.soundscape_mapping_small \
        --output_dir {output_dir} \
        --images_downloaded true \
        --output_name {region_name}_soundscape \
        --sat_type {sat_type} \
        --month {month} \
        --hour {hour} \
        --zoom_level {zoom_level} \
        --asource {asource} \
        --tsource {tsource} \
        --metadata_type {metadata_type} \
        --query_type {query_type} \
        --text_query_file {text_query_file} \
        --audio_query_file {audio_query_file} \
        --expr {expr}")

def main():
    # Use default values from demo_cfg dictionary
    log_dir = demo_cfg['log_dir']
    region_name = demo_cfg['region_name']
    latitude = demo_cfg['latitude']
    longitude = demo_cfg['longitude']
    spacing = demo_cfg['spacing']
    max_distance = demo_cfg['max_distance']
    sat_type = demo_cfg['sat_type']
    month = demo_cfg['month']
    hour = demo_cfg['hour']
    zoom_level = demo_cfg['zoom_level']
    asource = demo_cfg['asource']
    tsource = demo_cfg['tsource']
    metadata_type = demo_cfg['metadata_type']
    query_type = demo_cfg['query_type']
    text_query_file = demo_cfg['text_query_file']
    audio_query_file = demo_cfg['audio_query_file']
    expr = demo_cfg['expr']

    run_region_image_download(log_dir, region_name, latitude, longitude, spacing, max_distance)
    output_dir = os.path.join(log_dir, region_name)
    run_soundscape_mapping_small(output_dir, region_name, sat_type, month, hour, zoom_level,
                                 asource, tsource, metadata_type, query_type, text_query_file,
                                 audio_query_file, expr)

if __name__ == "__main__":
    main()
