import cv2
import os
from dipy.io.image import load_nifti, save_nifti
import nibabel as nib
import numpy as np
def img2nii(folder,image_prefix,save_nii,save_name,save_path):
    '''
    save_nii: save image to nii.gz
    save_path
    '''
    images = []
    for filename in np.sort(os.listdir(folder)):
        if (image_prefix in filename):
            if filename[-4:] == '.hdr':
                img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_ANYDEPTH)[:,:,0]
            else:
                img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
            images.append(img)
    images_xyz = np.array(images).swapaxes(0,1).swapaxes(1,2)
    #-----------------
    if np.max(images_xyz) > 1:
        if (image_prefix=='image'):
            images_xyz = images_xyz/255.0
        elif (image_prefix=='mask'):
            images_xyz = images_xyz /255
            images_xyz[images_xyz > 0.5] = 1
            images_xyz[images_xyz <= 0.5] = 0
    #-----------------
    img = nib.Nifti1Image(images_xyz, np.eye(4))
    if save_nii == True:
        img.header['xyzt_units'] = 11
        img.header['dim'][0] = 3#[3,128,128,14,1,1,0,0]
        img.header['dim'][4] = 1#[3,128,128,14,1,1,0,0]
        img.header['pixdim'] = [1,0.004,0.004,0.050,1,1,1,1]
        save_nifti(os.path.join(save_path,save_name+'.nii'), images_xyz.astype(np.float32), 
                   affine=np.eye(4),hdr=img.header)
    return images_xyz