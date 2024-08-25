Titanic Survival Prediction - Machine Learning Project
Overview
This repository contains the code and documentation for the Titanic Survival Prediction project. The project aims to build a robust machine learning model that predicts whether a passenger on the Titanic survived or not based on their demographic, socio-economic, and voyage-related features. The solution employs advanced ensemble learning techniques, including stacking classifiers, to achieve high predictive accuracy.

Table of Contents
Project Structure
Features
Data
Setup and Installation
Usage
Model Training
Evaluation
Results
Contributing
License
Project Structure


├── notebook
│   ├── titanic_project.ipynb   # Main Jupyter Notebook containing all steps
├── model_training.log          # Log file containing details of model training
├── final_submission.csv        # Final submission file for Kaggle
├── README.md                   # This file

Features
Feature Engineering: Custom features are engineered from the original dataset to improve model performance.
Model Training: Various machine learning models are trained, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and CatBoost.
Ensemble Learning: Stacking classifiers with different meta-learners (CatBoost, MLP, Logistic Regression) are used to improve prediction accuracy.
Hyperparameter Tuning: RandomizedSearchCV is utilized to find the optimal hyperparameters for the models.
Logging: Detailed logging is implemented to track the training process, including model performance metrics.

Data
The dataset used for this project is the Titanic dataset, available on Kaggle. It contains information about the passengers aboard the Titanic, including their age, gender, ticket class, fare, and whether they survived the disaster.

Setup and Installation
Prerequisites
Python 3.8 or higher
Jupyter Notebook
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/user-name/titanic-survival-prediction.git
cd titanic-survival-prediction
Install the required dependencies:

pip install -r requirements.txt
Launch Jupyter Notebook:

jupyter notebook
Open the Notebook:

In the Jupyter interface, navigate to the notebooks directory and open the titanic_project.ipynb file.

Usage
Explore the Data:

Open the titanic_project.ipynb notebook and run the cells in sequence to explore the data, perform feature engineering, and preprocess the data.

Train the Models:

Continue running the cells to train various machine learning models and evaluate their performance.

Generate Submission File:

The final predictions will be saved as final_submission.csv in the root directory.

Model Training
The training process involves:

Loading the preprocessed and engineered features.
Performing cross-validation on individual models.
Conducting hyperparameter tuning using RandomizedSearchCV.
Training stacking classifiers with different meta-learners.
Selecting the best-performing model based on cross-validation accuracy.
Evaluation
Each model is evaluated using cross-validation, and the best-performing model is selected based on accuracy. The selected model is then used to generate predictions on the test set.

Results
The final submission achieved an accuracy of approximately 78% on the test set, which was submitted to the Kaggle competition.

License
This project is licensed under the MIT License. See the LICENSE file for details.
