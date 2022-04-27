# Mini-Project 4: UNet Model

Team contributors: Zoé Ducroux, Daisy Jayson and Erjun Zhang


## Summary
U-Net is a classic deep learning method and it has been widely applied to image segmentation. It can be used for small dataset medical image segmentation and that data augmentation is the key to improve model performance. This project implemented the model from scratch following ideas in the [paper](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).  Then we pre-processed (augmented) the data and applied the model on dataset without augmentation and with different augmentation size. Finally, test accuracy of $81.40\%$ was achieved without data augmentation and accuracy of $86.02\%$ was achieved by augmenting the dataset to from $21$ to $147$ images. By virtually checking and comparing performance values, we concluded that U-Net model can do small dataset segmentation and data augmentation can improve model perfomances. It could be interesting for medical imaging researchers and data scientists. 

### Background


### Tools 

The mini-project will rely on the following technologies: 
 * Python to be the main language used to complete this project.
 * Tensorflow (keras)
 * DIPY



### Data 

This project used data from online dataset offered by:
1. Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC): dataset and the discription can be found [here: ISBI 2012 Segmentation Challenge](https://imagej.net/events/isbi-2012-segmentation-challenge).


## Results 
* Figure 1
* Figure 2
* The training weights can be [update! download here](https://imagej.net/events/isbi-2012-segmentation-challenge);
* The test image files and the corresponding segmentations can be [update! download here](https://imagej.net/events/isbi-2012-segmentation-challenge);

The report has been completed submitted to be reviewed; One the reviewing finished, we will be released here. 

## Dependencies

* Operation system: Linux ([Pop!_OS](https://pop.system76.com/) 20.10);
* Precessor: Intel® Core™ i7-9700KF CPU @ 3.60GHz × 8, 62.7 GiB memory;
* Python3 were used in this project;
* [Jupyter notebook](https://jupyter.org/) are needed to run jupyter notebooks in this project;
* [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) was used to view images;
* Python library details can be found in [requirements file](https://github.com/zhangerjun/UNet_model/blob/main/requirements.txt).



## Conclusion and acknowledgement

This work was inspired by the [publication](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/) of U-Net model.
This is the teamwork.
