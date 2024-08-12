#TSNE plot for embeddings saved using script val_embeds.py

import h5py as h5
import torch
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
import code
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def get_files(embeds_path):
    embeds = h5.File(embeds_path,"r")
    keys = np.array(embeds.get('keys'))
    sources = [str(key).replace("'","")[1:].split("-")[0] for key in keys]
    sat_embeds = torch.tensor(np.array(embeds.get('sat_embeds')))
    text_embeds = torch.tensor(np.array(embeds.get('text_embeds')))
    audio_embeds = torch.tensor(np.array(embeds.get('audio_embeds')))

    return sources, sat_embeds, text_embeds, audio_embeds



if __name__ == '__main__':
    
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--embeds_path', type=str, default='/storage1/fs1/jacobsn/Active/user_k.subash/projects/PSM_public/PSM/logs/results/embeds/val_embeds_GeoSound_pcmepp_metadata_sentinel-none-1.0-ZL-1-0.h5')

    args = parser.parse_args()
    sources, sat_embeds, text_embeds, audio_embeds = get_files(args.embeds_path)


    # Define color map for each source type
    color_map = {
        'yfcc': 'red',
        'aporee': 'blue',
        'iNat': 'green',
        'freesound': 'orange',
        # Add more colors as needed for additional source types
    }

    tsne = TSNE(n_components=2, random_state=42)
    audio_embeds_2d = tsne.fit_transform(audio_embeds)
    text_embeds_2d = tsne.fit_transform(text_embeds)
    sat_embeds_2d = tsne.fit_transform(sat_embeds)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each t-SNE result
    for ax, embeds_2d, title in zip(axs, [audio_embeds_2d, text_embeds_2d, sat_embeds_2d], ['Audio Embeddings', 'Text Embeddings', 'Overhead-Image Embeddings']):
        ax.set_title(title)
        for source_type in color_map:
            mask = np.array([source == source_type for source in sources])
            ax.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], color=color_map[source_type], label=source_type, alpha=0.2)
        ax.legend()

    plot_name = args.embeds_path.split("/")[-1].replace(".h5","_tsne.png")
    save_path = "/".join(args.embeds_path.split("/")[:-1])
    plot_path = os.path.join(save_path,plot_name)
    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()
