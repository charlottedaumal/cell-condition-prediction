# From Expression to Experiment: Predicting Cell Condition

**Authors:** Charlotte Daumal & Clemence Kiehl  
**Context:** This mini-project was completed as part of the *Introduction to machine learning for bioengineers* course taught by Prof. Johanni Brea at EPFL  
**Language:** Julia   
**Date:** December 2022  

---

## 📝 Project Description  

In an experiment on epigenetics and memory, Giulia Santoni (from the lab of Johannes Gräff at EPFL) measured the gene expression levels in multiple cells of a mouse brain under three different conditions that we call KAT5, CBP and eGFP. In this project, the goal is to predict – as accurately as possible – for each cell the experimental condition (_KAT5_, _CBP_ or _eGFP_) under which it was measured, given only the gene expression levels.

---

## 🧠 Methods & Models

To solve this classification task, we implemented and compared several linear and nonlinear models:

- Logistic Regression  
- Random Forest Classifier  
- Neural Network  
- K-Nearest Neighbors (KNN)

---

## 💻 Repository Structure

### 1) Code Scripts (in Julia `.jl`)
> `/src/`
- Data visualization  
- Data denoising  
- Linear model implementation  
- Nonlinear models implementation

### 2) Data Visualizations
> `/visualization/`
- PCA with feature vectors  
- 2D PCA plots for cluster identification

### 3) Model Evaluations
> `/models_assessment/`
- Proportion of variance explained by predictors  
- Confusion matrices for all models

### 4) Report
> `Final_Report.pdf`
- Full explanation of our pipeline:  
  data cleaning, visualization, model building, results, and conclusions

---

## 📝 Requirements

- All `.jl` scripts should be run from within the root of the repository

> **Note:** Some models (especially nonlinear ones) may take a long time to run depending on your machine.

---

## ⚙️ How to use it ?

1. Download the `.ZIP` files from our shared [Google Drive folder](https://drive.google.com/drive/folders/19PFEhJPrm3AXHuXpxQQnzs8CfaN4v8bU?usp=share_link)
2. Create a folder named `data/` at the root of the repository
3. Move the downloaded ZIP files into the `data/` folder
4. Unzip the files
5. Run any `.jl` file from the `/src/` directory to:
   - Visualize or denoise the data
   - Train and evaluate a model

---

## License

This project is for educational purposes.  
Content © Charlotte Daumal & Clemence Kiehl. Academic use only.
