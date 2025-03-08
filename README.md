# ML-Projects
Placement Prediction Project

This project develops a Machine Learning model to predict whether a student will get placed in a company based on academic and skill-based attributes. The model helps students assess their job eligibility and improve their chances.

Dataset and Goal

The dataset used is modified_placement_data.csv.  
The goal is to automate placement prediction using machine learning.

Model Used

The project applies Logistic Regression, which is best suited for binary classification problems such as determining whether a student will be placed or not.

Data Collection and Preprocessing

The dataset consists of student academic records and other related attributes.  
Before training the model, the data undergoes preprocessing:
- Handling missing values using dropna().
- Selecting relevant features like CGPA, IQ, and other academic scores.
- Performing Exploratory Data Analysis (EDA) to understand patterns and relationships.

Machine Learning Model

The training process involves:
- Splitting data into training and testing sets using train_test_split().
- Training the model using LogisticRegression().fit(X_train, y_train).
- Making predictions using model.predict(X_test).

Performance Evaluation

To measure the model's effectiveness, different evaluation metrics are used:
- Accuracy Score to check overall correctness.
- Confusion Matrix to analyze correct and incorrect predictions.
- Precision and Recall to evaluate false positives and false negatives.

How to Run

1. Install the required libraries using the command:
   
   pip install numpy pandas matplotlib seaborn scikit-learn

2. Ensure the dataset file modified_placement_data.csv is in the same directory.
3. Open Project3-1.ipynb in Jupyter Notebook and run all cells.

Future Improvements

To enhance the project, the following improvements can be made:
- Adding more features like extracurricular activities and internships.
- Testing other machine learning models like Decision Trees or Random Forest.
- Improving data quality for better prediction accuracy.
