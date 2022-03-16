# Mercedes Benz Mobility - Interview Task
---
## _Image Segmentation for Car Damage_

> Create a jupyter notebook to train a segmentation model for damage detection based on the given training data. 
> Document your approach, requirements, results and valuable insights (ReadMe).
> The solution should be properly documented in your personal GitHub-Account.

- Type some Markdown on the left
- See HTML in the right
- ✨Magic ✨
---
## Table of Contents
- [Approach](#approach)
- [Challenges](#challenges)
- [Libraries used](#libraries-used)
- [Results](#results)

## Approach

0. Understand the relevance and business impact of this task.
1. Understand the data, get a feeling for the training data and what to predict.
2. Prepare the data to be used in a machine learning model.
3. Train a machine learning model and finetune it.
4. Evaluate the results.



## Challenges
This project posed a few challenges.
  
- Lack of training images.
- Lack of computing power. 
- Time constraints.

## Libraries used
- openCV
- pandas
- keras / tensorflow
- imgaug (imgage augmentation)
- matplotlib

## Results



#### Example predictions
| Input  | Prediction  |
|---|---|
|![Input Image](ressources\screenshots\img.jpg)   | ![Overlayed Mask](ressources\screenshots\sample_prediction_10.png)  |
|![Input Image](ressources\screenshots\1.jpg) | ![Overlayed Mask](ressources\screenshots\pred2.png) |
|![Input Image](ressources\screenshots\img.png) |![Overlayed Mask](ressources\screenshots\pred.png) |

