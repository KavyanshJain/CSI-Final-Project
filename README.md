# Creditworthiness Prediction Project

## Overview
This project predicts the creditworthiness of individuals using the Statlog (German Credit Data) dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). It uses a random forest classifier to predict whether an individual is creditworthy (good credit) or not (bad credit), incorporating a cost matrix where misclassifying bad credit as good costs 5, and good as bad costs 1. The project includes exploratory data analysis (EDA), preprocessing, model training, a Streamlit web application for predictions, and a PDF report summarizing the findings.

---
## Live webapp link:
https://creditworthiness-predictor.streamlit.app/

---
## Webapp Screenshot:

<img width="1366" height="2311" alt="image" src="https://github.com/user-attachments/assets/d5fc89f8-fdb7-4ed5-8f0e-36ed08a1b293" />


---
## Repository Contents
- **creditworthiness_project.ipynb**: Jupyter Notebook containing the full workflow, including data loading, detailed EDA, preprocessing, outlier handling, feature selection, and random forest model training.
- **app.py**: Streamlit web application for inputting financial attributes and predicting creditworthiness using the trained model.
- **report.pdf**: Report with EDA insights, preprocessing details, model performance, and conclusions.
- **credit_model.pkl**: (Generated after running the notebook saved in `model/`) The trained random forest model and preprocessing pipeline.
- **german.data**: (Required, not included) The dataset file, downloadable from the UCI repository.

## Requirements
To run the project, install the following Python libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit joblib
```
---

## Setup and Usage
1. **Clone the repoistory**:
   ```bash
   git clone https://github.com/KavyanshJain/CSI-Final-Project.git

   ```
2. **Run the Jupyter Notebook**:
   - Open `creditworthiness_project.ipynb` in Jupyter Notebook.
   - Execute all cells to perform EDA, preprocess the data, train the random forest model, and save the model as `credit_model.pkl`.
   - The notebook includes detailed visualizations (histograms, boxplots, violin plots, stacked bar plots) and insights for each step.

3. **Run the Streamlit App**:
   - Ensure `credit_model.pkl` is created before running`app.py`.
   - Run the app using:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL in a browser to input financial attributes and get creditworthiness predictions.

---

## Project Details
- **Dataset**: The Statlog (German Credit Data) dataset contains 1000 instances with 20 attributes (7 numerical, 13 categorical) and a binary target (1=good credit, 2=bad credit).
- **EDA**: Includes distribution analysis, outlier detection, correlation analysis, and relationships between features and the target variable.
- **Preprocessing**: Handles outliers via log transformation for `credit_amount` and `duration`, uses ordinal encoding for ordinal variables (e.g., `employment`), and one-hot encoding for nominal variables (e.g., `purpose`).
- **Model**: A random forest classifier with class weights `{0:1, 1:5}` to account for the cost matrix, optimized using grid search for F1-score.
- **Feature Selection**: Identifies low-importance features but retains all for robustness.
- **Streamlit App**: Provides a user-friendly interface with dropdowns and numerical inputs for predicting creditworthiness.
- **Report**: Summarizes EDA insights, preprocessing steps, model performance, and conclusions in a professional PDF format.

---

## Notes
- Ensure `german.data` is in the correct directory `data/` before running the notebook.
- The Streamlit app requires `credit_model.pkl`, which is generated by the notebook and it will be stored in `model`.

---
