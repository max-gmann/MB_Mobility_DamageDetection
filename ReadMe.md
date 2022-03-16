# Mercedes Benz Mobility - Interview Task
---
## _Image Segmentation for Car Damage_

> Create a jupyter notebook to train a segmentation model for damage detection based on the given training data. 
> Document your approach, requirements, results and valuable insights (ReadMe).
> The solution should be properly documented in your personal GitHub-Account.

---
## Table of Contents
- [Approach](#approach)
- [Libraries used](#libraries-used)
- [Results](#results)
---
## Approach

0. Understand the relevance and business impact of this task.
1. Understand the data, get a feeling for the training data and what to predict.
2. Prepare the data to be used in a machine learning model.
3. Train a machine learning model and finetune it.
4. Evaluate the results.

## Libraries used
- openCV
- pandas
- keras / tensorflow
- imgaug (imgage augmentation)
- matplotlib


## Model Training
For the image segmentation task I used a convolutional neural network with a UNET architecture as implemented [here](https://wandb.ai/ayush-thakur/image-segmentation/reports/Image-Segmentation-Using-Keras-and-W-B--VmlldzoyNTE1Njc).
Training the model posed a few challenges:
- Small number of training images.
- Imbalanced classes 
- Computationally expensive
The model was trained for 100 epochs on a Google Colab GPU instance and an image size of 256x256. 
![Loss](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/loss.png)
![Accuracy](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/accuracy.png)
The loss seems to plateau after epoch 60. Further tuning of the optimizer, loss function and data augmentation could improve the results.
The accuray of 94% is not very meaningful due to the imbalanced nature of this segmentation task.

## Results

Given to the nature of such a small training data set and time restraints of this project I'm satisfied with the model's performance.
It sometimes does suprisingly well in following complex shapes. For some types of damages, however, it seems to default to a square block in the center of the image. It also appears to have problems with multiple horizontal scratches. Possible reasons for this include:
- the model wasn't trained long enough, as the training loss and validation loss have not converged yet.
- the model is very large and there are too few training images. Big deep learning models require huge amounts of data to deliver accurate results. While data augmentation expanded the training data to around 300 images, this is most likely still not enough.
- choice of loss function. Since the training data is highly imbalanced a different loss function such as Dice coefficent could yield better results.
- unsuitable data augmentation. This aspect of data preparation can be utilized more by creating more variations with additional transformations. 

To further improve the model's performance, these would be my next steps:
- Get more data. 
- Implement Transfer Learning with a pre-trained model. 
- Experiment with different model architectures and loss functions.
- Experiment with different augmentation methods and image sizes.
- Modularize the training and data loading process so the pipeline can be automated.

#### Example predictions
| Input  | Prediction  |
|---|---|
|![Input Image](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/img.jpg)   | ![Overlayed Mask](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/sample_prediction_10.png)  |
|![Input Image](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/1.jpg) | ![Overlayed Mask](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/pred2.png) |
|![Input Image](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/img.png) |![Overlayed Mask](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/pred.png) |

#### Test Images
![Test 1](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/sample_prediction_0.png)
![Test 2](https://github.com/max-gmann/MB_Mobility_DamageDetection/blob/main/ressources/screenshots/sample_prediction_1.png)
---
