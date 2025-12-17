# SDSS-Galaxy-Classification
# 🌌 SDSS Galaxy Classification using Machine Learning

This project implements a **Machine Learning–based Galaxy Classification System** using data from the **Sloan Digital Sky Survey (SDSS)**.  
The system classifies galaxies into two categories:

- **Starforming**
- **Starbursting**

A trained ML model is integrated with a **Flask web application**, allowing users to input astronomical parameters and obtain real-time predictions through a simple web interface.

---

## 📌 Project Objectives

- To analyze SDSS galaxy data and identify key features influencing galaxy formation.
- To build a robust classification model using Machine Learning.
- To deploy the trained model using a Flask web application.
- To provide an interactive UI for real-time galaxy classification.

---

## 🧠 Machine Learning Workflow

1. **Data Loading**
   - SDSS galaxy dataset (`sdss_100k_galaxy_form_burst.csv`)

2. **Data Preprocessing**
   - Handling missing values
   - Replacing invalid values (e.g. `-9999`)
   - Encoding categorical variables
   - Outlier treatment using IQR
   - Feature scaling

3. **Feature Selection**
   - Used `SelectKBest` with `f_classif`
   - Selected top 10 most relevant features

4. **Class Imbalance Handling**
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)**

5. **Model Training**
   - Algorithms tested:
     - Decision Tree
     - Logistic Regression
     - Random Forest
   - **Random Forest Classifier** selected as final model due to superior performance

6. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

7. **Model Saving**
   - Final model saved as `RF.pkl`

---

## 🧪 Selected Features Used for Prediction

The model uses the following 10 features:

- `i`
- `z`
- `modelFlux_z`
- `petroRad_g`
- `petroRad_r`
- `petroFlux_z`
- `petroR50_u`
- `petroR50_g`
- `petroR50_i`
- `petroR50_r`

---

## 🌐 Web Application (Flask)

- Built using **Flask**
- Two-page workflow:
  - **Input Page** → User enters galaxy parameters
  - **Result Page** → Displays predicted galaxy class
- Same CSS styling used across pages
- Flask runs locally using Jupyter Notebook

---

## 🗂️ Project Structure

