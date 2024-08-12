#Given a location (lat,long), this script downloads images for a bounding box of size 10 km x 10 km centered around the given location.
import os
import pandas as pd
from tqdm import tqdm
import urllib.request
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from ..config import cfg

bingmap_api = cfg.bingmap_api_key1


# Function to calculate bounding box
def get_bounding_box(lat, lon, max_distance=10):
    deg_distance = max_distance / 111
    min_lat = lat - deg_distance/2
    max_lat = lat + deg_distance/2
    min_lon = lon - deg_distance/2
    max_lon = lon + deg_distance/2
    return [min_lon, min_lat, max_lon, max_lat]

# Function to generate grid points
def generate_grid(lat, lon, radius=125, max_distance=10):
    bbox = get_bounding_box(lat, lon, max_distance=max_distance)
    deg = radius * 2 / 111000
    x = np.arange(bbox[0] + deg / 2, bbox[2], deg)
    y = np.arange(bbox[1] + deg / 2, bbox[3], deg)
    xv, yv = np.meshgrid(x, y)
    coords = np.c_[xv.flatten(), yv.flatten()]
    return coords

# Function to download images
def download_images(region_file_path, output_dir):
    zoom_level = 18
    im_size = "1500,1500"
    template_url = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%s/%d?mapSize=%s&key=%s" % ("%s", zoom_level, im_size, bingmap_api)
    region_df = pd.read_csv(region_file_path)
    image_paths = []
    print("Downloading images for the region...")
    success = 0
    for i in tqdm(range(len(region_df))):
        sample = region_df.iloc[i]
        tmp_loc = "%s,%s" % (sample.latitude, sample.longitude)
        image_url = template_url % tmp_loc
        image_file = os.path.join(output_dir, f"{sample.id}_{sample.latitude}_{sample.longitude}.jpeg")
        image_paths.append(image_file)
        try:
            urllib.request.urlretrieve(image_url, image_file)
            success += 1
        except urllib.error.HTTPError as e:
            print("HTTP Error:", e.code)
            continue
        except Exception as e:
            print("Other Error:", e)
            continue
    print(f"{success} images downloaded successfully.")
    return image_paths

if __name__ == "__main__":
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--log_dir', type=str, default='/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/maps', help='Directory to save the grid points and images')
    parser.add_argument('--region_name', type=str, default='dummy', help='Name of the region')
    parser.add_argument('--latitude', type=float, default=40.7128, help='Latitude of the input location')
    parser.add_argument('--longitude', type=float, default=-74.0060, help='Longitude of the input location')
    parser.add_argument('--spacing', type=int, default=125, help='Spacing between grid points')
    parser.add_argument('--max_distance', type=int, default=5, help='Maximum distance from the input location in km')
    args = parser.parse_args()

    # Generate grid
    grid = generate_grid(args.latitude, args.longitude, radius=args.spacing / 2, max_distance=args.max_distance)
    df = pd.DataFrame(grid, columns=['longitude', 'latitude'])
    df['id'] = range(len(df))

    # Create directory if not exists
    output_dir = os.path.join(args.log_dir, args.region_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save grid points to CSV
    grid_csv_path = os.path.join(output_dir, f'grid_points_{args.region_name}.csv')
    df.to_csv(grid_csv_path, index=False)
    print(f"Grid points saved to {grid_csv_path}")

    # Download images
    image_paths = download_images(grid_csv_path, output_dir)
    print(f"Region images downloaded to {output_dir}")
