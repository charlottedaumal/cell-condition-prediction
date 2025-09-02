# Prediction of Cell Condition

**Authors:** Charlotte Daumal & Clemence Kiehl
**Context:** This project was completed as part of the *Introduction to machine learning for bioengineers* course taught by Prof. Johanni Brea
**Language:** Julia  
**Date:** December 2022

---

## 📝 Project Description  

In an experiment on epigenetics and memory, Giulia Santoni (from the lab of Johannes Gräff at EPFL) measured the gene expression levels in multiple cells of a mouse brain under three different conditions that we call KAT5, CBP and eGFP. In this project, the goal is to predict – as accurately as possible – for each cell the experimental condition (KAT5, CBP or eGFP) under which it was measured, given only the gene expression levels.

---

## Methods used to build our models

We used several linear and nonlinear methods to achieve our results, such as :

* Logistic regression
* Random Forest Classification
* Neural Network
* Simple KNN Classifier

---

## Organisation

One can find our mini project under the *ClemenceKiehl/Epigenetics_and_Memory_ML_Project* repository. It contains:

* a folder **src** that contains the code in the format of .jl scripts, more specifically the code for:
  * visualization of the data.
  * denoising the data.
  * a model based on a linear method.
  * models based nonlinear methods.
* a folder **visualization** that contains the different visualizations of the data, more specifically:
  * a PCA visualization of the data with the vectors.
  * a PCA visualization of the data in 2D to identify potential clusters.
* a folder **models_assessment** that contains some evaluations of our models, more specifically:
  * graphs of the proportion of the variance and tcumulative proportion of variance explained by data's predictors.
  * confusion matrices for all our models.
* a report which explains how we cleaned/visualized the data, which linear and nonlinear methods we implemented and the results we got.

---

## Requirements

* Use a Julia environment **v1.7**.
* In order to run our .jl scripts, please be in the repository *ClemenceKiehl/Epigenetics_and_Memory_ML_Project*. The different results can be reproduced by running the .jl scripts in the repository src.  

_Note_: depending on the model, it can take very long to get a result.

---

## Instructions to run our code

1. Download the .ZIP files available in this repository on Google Drive : https://drive.google.com/drive/folders/19PFEhJPrm3AXHuXpxQQnzs8CfaN4v8bU?usp=share_link
2. Place these files in a folder named "data" in the GitHub repository on your computer.
3. Unzip the .ZIP files.
4. Run any file from the src folder to either visualize the data or run specific models based on linear or nonlinear methods.

---

## License

This project is for educational purposes.  
Content © Charlotte Daumal & Clemence Kiehl. Academic use only.
