# Epigenetics and Memory ML Project
The goal of our Machine Learning mini project is to predict under which experimental conditions different cell types were measured, given their gene expression level.

## Requirements

Use a Julia environment v1.7. 
The package Pkg is essential to run all the code scripts and to use the MLCourse related packages.

## Methods used to build models

We used several linear and nonlinear method to achieve our results, such as :

* Logistic regression
* Random Forest Classification
* Neural Network
* Simple KNN Classifier

## Organisation

One can find our mini project under the *ClemenceKiehl/Epigenetics_and_Memory_ML_Project* repository. It contains:

* a folder **src** that contains the code in the format of .jl scripts, more specifically the code for:
  * visualization of the data.
  * denoising the data.
  * a model based on a linear method.
  * models based non-linear methods.
* a folder **visualization** that contains the different visualizations of the data, more specifically:
  * a PCA visualization of the data with the vectors.
  * a PCA visualization of the data in 2D to identify potential clusters.
  * a PCA visualization of the data in 3D to identify potential clusters.
* a folder **models_assessment** that contains some evaluations of our models, more specifically:
  * graphs of the proportion of the variance and tcumulative proportion of variance explained by data's predictors.
  * confusion matrices for all our models.
* a report which explains how we cleaned/visualized the data, which linear and nonlinear methods we implemented and the results we got.

## Instructions to run our code

1. Download the .ZIP files available in this repository on Google Drive : https://drive.google.com/drive/folders/19PFEhJPrm3AXHuXpxQQnzs8CfaN4v8bU?usp=share_link
2. Place these files in a folder named "data" in the GitHub repository on your computer.
3. Run any file from the src folder to eitehr visualize the data or run specific linear or non-linear models.

## Team Members

* Charlotte Daumal
* Clémence Kiehl
