def img2nii(folder,image_prefix,save_nii,save_name,save_path):
    '''
    save_nii: save image to nii.gz
    save_path
    '''
    import cv2
    import os
    from dipy.io.image import load_nifti, save_nifti
    import nibabel as nib
    import numpy as np
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


def ImageGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,
                   image_color_mode,mask_color_mode,
                   image_save_prefix,mask_save_prefix,
                    flag_multi_class,num_class,
                   save_to_dir,target_size,seed):
    '''
    To generate image and mask
    if you want to visualize the results of generator, set save_to_dir = "your path"
    #---------------------------------------------------------------------------------
    Example of usage:
    # Data augmentation
    data_gen_args = dict(rotation_range=70,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        fill_mode='nearest')
    Image_generated = ImageGenerator(batch_size=10,train_path='data/membrane/train',
                   image_folder='image',mask_folder='label',aug_dict=data_gen_args,
                   image_color_mode = "grayscale",mask_color_mode = "grayscale",
                   image_save_prefix  = "image",mask_save_prefix  = "mask",
                         flag_multi_class = False,num_class = 2,
                   save_to_dir = "data/membrane/train/aug",target_size=(388,388),seed = 1)
    num_batch = 30
    for i,batch in enumerate(Image_generated):
        if(i >= num_batch):
            break
    #---------------------------------------------------------------------------------
    '''
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np 
    import os
    import glob
    import skimage.io as io
    import skimage.transform as trans
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle=True,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=True,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        yield (img,mask)
# Function definition
def imgNormlize(folder,image_prefix,mask,save_nii,save_name,save_path):
    '''
    normalize images
    '''
    import cv2
    import os
    from dipy.io.image import load_nifti, save_nifti
    import nibabel as nib
    import numpy as np
    images = []
    for filename in np.sort(os.listdir(folder)):
        if image_prefix in filename:
            img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
            if np.max(img) > 1:
                if (img is not None) and (mask==False):
                    img = img/255.0
                elif (img is not None) and (mask==True):
                    img = img /255
                    img[img > 0.5] = 1
                    img[img <= 0.5] = 0
            images.append(img)
    images_xyz = np.array(images).swapaxes(0,1).swapaxes(1,2)
    img = nib.Nifti1Image(images_xyz, np.eye(4))
    if save_nii == True:
        img.header['xyzt_units'] = 11
        img.header['dim'][0] = 3#[3,128,128,14,1,1,0,0]
        img.header['dim'][4] = 1#[3,128,128,14,1,1,0,0]
        img.header['pixdim'] = [1,0.004,0.004,0.050,1,1,1,1]
        save_nifti(os.path.join(save_path,save_name+'.nii'), images_xyz.astype(np.float32), 
                   affine=np.eye(4),hdr=img.header)
    return images_xyz
# Function definition
def imgPadding(folder,mode,size,save_path):
    '''
    padding images
    '''
    import cv2
    import os
    from dipy.io.image import load_nifti, save_nifti
    import nibabel as nib
    import numpy as np
    for filename in np.sort(os.listdir(folder)):
        if filename[-4:] == '.hdr':
            img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_ANYDEPTH)[:,:,0]
        else:
            img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        px = int((size[0]-img.shape[0])/2)
        py = int((size[1]-img.shape[1])/2)
        img_padded = np.pad(img, (px,py), mode)
        cv2.imwrite(os.path.join(save_path,filename), img_padded)
# Function definition
def array2nii(array_input,save_name,save_path):
    '''
    save array to nifti file
    '''
    import cv2
    import os
    from dipy.io.image import load_nifti, save_nifti
    import nibabel as nib
    import numpy as np
    images_xyz = array_input
    img = nib.Nifti1Image(images_xyz, np.eye(4))
    save_nii = True
    if save_nii == True:
        img.header['xyzt_units'] = 11
        img.header['dim'][0] = 3#[3,128,128,14,1,1,0,0]
        img.header['dim'][4] = 1#[3,128,128,14,1,1,0,0]
        img.header['pixdim'] = [1,0.004,0.004,0.050,1,1,1,1]
        save_nifti(os.path.join(save_path,save_name+'.nii'), images_xyz.astype(np.float32), 
                   affine=np.eye(4),hdr=img.header)