# Mini-Project 4: UNet Model

Team contributors: Zoé Ducroux, Daisy Jayson and Erjun Zhang


## Summary
U-Net is a classic deep learning method and it has been widely applied to image segmentation. It can be used for small dataset medical image segmentation and that data augmentation is the key to improve model performance. This project implemented the model from scratch following ideas in the [paper](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).  Then we pre-processed (augmented) the data and applied the model on dataset without augmentation and with different augmentation size. Finally, test accuracy of 81.40\% was achieved without data augmentation and accuracy of 86.02\% was achieved by augmenting the dataset from 21 to 147 images. By virtually checking and comparing performance values, we concluded that U-Net model can do small dataset segmentation and data augmentation can improve model perfomances. It could be interesting for medical imaging researchers and data scientists. 

## Background
U-Net is a classic deep learning method and it has been widely applied to image segmentation. 
This model has been developed since 2012 and has many variants, but we hardly found code (Python) to reproduce the results of this paper. Thus, we set the project goal as reproducing the U-Net architecture for image segmentation following the idea from this original publication and explore data augmentation effects on its perfomance. 

## Method

### Data 

Dataset used in this project:
* Transmission Electron Microscopy data set of the Drosophila first instar larva ventral nerve cord; dataset and its discription can be found [here](https://imagej.net/events/isbi-2012-segmentation-challenge).

Example images and labels:
* ![U-Net can work](https://github.com/zhangerjun/UNet_model/blob/main/results/Fig1.png)

### Experimental setup
Experiments designed and operations we used were to meet our two general project goal. To see if claims of the author are correct or not:
* we first implemented the model and applied it on the chosen dataset, then tested if it can segment the image well;
* model performance on dataset without data augmentation and with small data augmentation would be compared to see if augmented data can still be used as training dataset;
* model performance on dataset with different augmented dataset would be used to explore the behavior of the model performance while varying augmentation (from no augmentation to 147 augmented samples).

* ![U-Net can work](https://github.com/zhangerjun/UNet_model/blob/main/results/Fig2.png)

## Results 
* ![U-Net can work](https://github.com/zhangerjun/UNet_model/blob/main/results/Fig3.png)
* ![U-Net can work](https://github.com/zhangerjun/UNet_model/blob/main/results/Fig4.png)
* ![U-Net can work](https://github.com/zhangerjun/UNet_model/blob/main/results/Fig5.png)
* ![U-Net can work](https://github.com/zhangerjun/UNet_model/blob/main/results/Fig6.png)
* The training weights can be [download here](https://github.com/zhangerjun/UNet_model/blob/main/results/Test_images_and_predictions.zip);
* The test image files and the corresponding predicted segmentations can be [download here](https://github.com/zhangerjun/UNet_model/blob/main/results/Test_images_and_predictions.zip).
* The report can be found and [download here](https://github.com/zhangerjun/UNet_model/reports/Project4_report.pdf);


## Reproducibility
### Dependencies

* Operation system: Linux ([Pop!_OS](https://pop.system76.com/) 20.10);
* Precessor: Intel® Core™ i7-9700KF CPU @ 3.60GHz × 8, 62.7 GiB memory;
* Python3 was used in this project;
* [Jupyter notebook](https://jupyter.org/) are needed to run jupyter notebooks in this project;
* [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) was used to view images;
* Python library details can be found in [requirements file](https://github.com/zhangerjun/UNet_model/blob/main/requirements.txt).

### Usage
* Put training images and the corresponding labels into folder 'image0' and 'label0' (now the shape of input images have to be 512x512);
* The feed-in data will be split into training set and test set wit ratio 7/3 autmatically; 
* Run 'UNet_without_data_augmentation.ipynb' to test if model can work on small dataset segmentation;
* Run 'UNet_effect_by_augmentation.ipynb' to do augemnatation experiment.


## Conclusion and acknowledgement
U-Net model can do small dataset segmentation and data augmentation can improve model perfomances.

In the future, we will try to make it work with any shape of imput images and also multi-class segmentation will be added.

This work was inspired by the [publication](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/). It was completed in NeuroPoly at Polytechnique Montreal and Magic lab in TransMedTech Institute in CHU Sainte Justine Hospital.
