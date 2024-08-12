#Taken from https://github.com/sustainlab-group/SatMAE/blob/main/util/datasets.py

import torch
from torchvision import transforms
from PIL import Image


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tile_size = {'sentinel':256, 'bingmap':300, 'googleEarth':256} 
sat_gsd = {'sentinel':10, 'bingmap':0.6, 'googleEarth':0.2}


def zoom_transform(zoom_level=1, sat_type='sentinel'):
    interpol_mode = transforms.InterpolationMode.BICUBIC
    t = []
    input_size = zoom_level*tile_size[sat_type]
    t.append(transforms.ToTensor())
    t.append(transforms.CenterCrop(input_size))
    t.append(
        transforms.Resize(tile_size[sat_type], interpolation=interpol_mode,antialias=True), 
    )
    return transforms.Compose(t)


def sat_transform(is_train, input_size):
    """
    Builds train/eval data transforms for the dataset class.
    :param is_train: Whether to yield train or eval data transform/augmentation.
    :param input_size: Image input size (assumed square image).
    :param mean: Per-channel pixel mean value, shape (c,) for c channels
    :param std: Per-channel pixel std. value, shape (c,)
    :return: Torch data transform for the input image before passing to model
    """
    
    interpol_mode = transforms.InterpolationMode.BICUBIC

    t = []
    if is_train:
        t.append(transforms.ToTensor())
        t.append(
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode, antialias=True),
        )
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
    else:
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(input_size, interpolation=interpol_mode,antialias=True),
        )
        t.append(transforms.CenterCrop(input_size))
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


if __name__ == '__main__':
    import os
    from torchvision.io import read_image
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    for level in range(1,6):
        print("FOR ZOOM LEVEL:",level)
        sat_image_path = '/storage1/fs1/jacobsn/Active/user_k.subash/data_archive/aporee/images/bingmap'
        demo_image  =  os.listdir(sat_image_path)[0]

        #define transforms
        zoom_tr = zoom_transform(zoom_level=level, sat_type='bingmap')
        sat_tr = sat_transform(is_train=False, input_size=224)
        
        #read_image
        sat_img = read_image(os.path.join(sat_image_path,demo_image))
        sat_img = np.array(torch.permute(sat_img,[1,2,0]))
        plt.imshow(sat_img)
        plt.savefig("/storage1/fs1/jacobsn/Active/user_k.subash/projects/improved-PSM/sat2sound/logs/demo_image.jpeg")
        print("Original image",sat_img.shape)

        #get image for a given zoom level
        level_image = zoom_tr(sat_img)
        print("Image after zoom transform", level_image.shape)
        level_image = np.array(torch.permute(level_image,[1,2,0]))
        plt.imshow(level_image)
        plt.savefig("/storage1/fs1/jacobsn/Active/user_k.subash/projects/improved-PSM/sat2sound/logs/demo_level"+str(level)+".jpeg")
        
        #get image for resized one
        final_image = invTrans(sat_tr(level_image))
        print("Image after sat transform",final_image.shape)
        final_image = np.array(torch.permute(final_image,[1,2,0]))
        print(final_image.shape)
        plt.imshow(final_image)
        plt.savefig("/storage1/fs1/jacobsn/Active/user_k.subash/projects/improved-PSM/sat2sound/logs/demo_final_level"+str(level)+".jpeg")
        import code;code.interact(local=dict(globals(), **locals()));
   

