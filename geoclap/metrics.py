import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import os
import code

# Taken from: https://github.com/naver-ai/pcmepp/blob/main/pcmepp/evaluation.py
def compute_matmul_sims(images, captions, to_double=True):
    if to_double:
        images = images.astype(np.float64)
        captions = captions.astype(np.float64)
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def compute_csd_sims(images, captions, image_sigmas, caption_sigmas, to_double=True):
    """ Compute similarity scores while preventing OOM
    """
    csd = 2 - 2 * compute_matmul_sims(images.cpu().numpy().copy(), captions.cpu().numpy().copy(), to_double=to_double)
    image_sigmas = torch.exp(image_sigmas).cpu().numpy().copy()
    caption_sigmas = torch.exp(caption_sigmas).cpu().numpy().copy()
    if to_double:
        image_sigmas = image_sigmas.astype(np.float64)
        caption_sigmas = caption_sigmas.astype(np.float64)
    csd = csd + image_sigmas.sum(-1).reshape((-1, 1))
    csd = csd + caption_sigmas.sum(-1).reshape((1, -1))
    return -csd


def get_retrevial_metrics(modality1_emb, modality2_emb, normalized=False,k=100, loss_type="infonce"): #Used during training where we don't need dataframe of retrevial saved.
    if not normalized:
        # Normalize embeddings using L2 normalization
        modality1_emb_mean = normalize(modality1_emb['mean'], p=2, dim=1)
        modality2_emb_mean = normalize(modality2_emb['mean'], p=2, dim=1)
    else:
        modality1_emb_mean = modality1_emb['mean']
        modality2_emb_mean = modality2_emb['mean']

    if loss_type == "infonce":
        # Compute cosine similarity between embeddings
        cos_sim = torch.matmul(modality1_emb_mean, modality2_emb_mean.t()).detach().cpu().numpy() 
        distance_matrix = cos_sim
    else:
        distance_matrix = compute_csd_sims(modality1_emb_mean, modality2_emb_mean, modality1_emb['std'], modality2_emb['std'])
    K = distance_matrix.shape[0]
    # Evaluate Img2Sound
    results = []
    for i in list(range(K)):
        tmpdf = pd.DataFrame(dict(
            dist = distance_matrix[i,:]
        ))

        tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
        res = dict(
            rank=tmpdf.iloc[i]['rank']
        )
        results.append(res)
    df = pd.DataFrame(results)
    topk_str =str(1*k) 
    i2s_metrics = {
        'R@'+topk_str: (df['rank'] < k).mean(),
        'Median Rank': df['rank'].median(),
    }

    return i2s_metrics


def get_retrevial(modality1_emb, modality2_emb, keys,normalized=False,k=100,save_top=5, loss_type = "infonce"):
    
    if not normalized:
        # Normalize embeddings using L2 normalization
        modality1_emb_mean = normalize(modality1_emb['mean'], p=2, dim=1)
        modality2_emb_mean = normalize(modality2_emb['mean'], p=2, dim=1)
    else:
        modality1_emb_mean = modality1_emb['mean']
        modality2_emb_mean = modality2_emb['mean']

    if loss_type == "infonce":
        # Compute cosine similarity between embeddings
        cos_sim = torch.matmul(modality1_emb_mean, modality2_emb_mean.t()).detach().cpu().numpy() 
        distance_matrix = cos_sim
    else:
        modality1_emb_std_norm = torch.sqrt(torch.exp(modality1_emb['std'])).sum(dim=-1)
        modality2_emb_std_norm = torch.sqrt(torch.exp(modality2_emb['std'])).sum(dim=-1)
        distance_matrix = compute_csd_sims(modality1_emb_mean, modality2_emb_mean, modality1_emb['std'], modality2_emb['std'])
    
    K = distance_matrix.shape[0]
    
    # Evaluate Img2Sound
    results = []
    df_final = pd.DataFrame(columns=['key','top_keys''key_norm_std'])
    df_final['key'] = keys
    if loss_type == "pcmepp":
        keys_norm_std = modality1_emb_std_norm.detach().cpu().numpy()
        df_final['key_norm_std'] = keys_norm_std
    
    results_keys = []
    for i in list(range(K)):
        top_keys = []
        tmpdf = pd.DataFrame(dict(
            dist = distance_matrix[i,:]
        ))
        row_similarity = list(distance_matrix[i, :])
        top_indices = np.array(row_similarity).argsort()[-save_top:][::-1]
        top_keys = [keys[indice] for indice in top_indices]
        results_keys.append(top_keys)
        tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
        res = dict(
            rank=tmpdf.iloc[i]['rank']
        )
        results.append(res)
    df = pd.DataFrame(results)
    topk_str =str(1*k)
    if loss_type == "infonce":
        i2s_metrics = {
            'R@'+topk_str: (df['rank'] < k).mean(),
            'Median Rank': df['rank'].median(),
        }
    else:
        i2s_metrics = {
            'R@'+topk_str: (df['rank'] < k).mean(),
            'Median Rank': df['rank'].median(),
            'modality1_emb_std_norm_mean': modality1_emb_std_norm.mean(),
            'modality2_emb_std_norm_mean': modality2_emb_std_norm.mean(),
        }
        
    df_final['top_keys'] = results_keys
    return i2s_metrics, df_final


if __name__ == '__main__':
    def l2normalize(batch_embeddings):
        return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)
    
    modality1_embed_dict ={}
    modality1_embed_dict['mean'] = l2normalize(torch.randn(50,1024))
    modality1_embed_dict['std'] = torch.randn(50,1024)
    
    modality2_embed_dict ={}
    modality2_embed_dict['mean'] = l2normalize(torch.randn(50,1024))
    modality2_embed_dict['std'] = torch.randn(50,1024)
    keys = list(range(modality2_embed_dict['mean'].shape[0]))
    print("INFONCE LOSS")
    print(get_retrevial(modality1_embed_dict, modality2_embed_dict, keys, normalized=True,k=5, loss_type="infonce")[0])
    print("PCMEPP LOSS")
    print(get_retrevial(modality1_embed_dict, modality2_embed_dict, keys, normalized=True,k=5, loss_type="pcmepp")[0])

    #evaluate without dataframe
    print("INFONCE LOSS")
    print(get_retrevial_metrics(modality1_embed_dict, modality2_embed_dict, normalized=True,k=5, loss_type="infonce"))
    print("PCMEPP LOSS")
    print(get_retrevial_metrics(modality1_embed_dict, modality2_embed_dict, normalized=True,k=5, loss_type="pcmepp"))