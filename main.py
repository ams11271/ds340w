from data_loader import load_data
from feature_engineering import create_features
from models import train_logistic_model, train_random_forest
from evaluationmetrics import evaluate_model

# Define paths and target
file_path = "Healthcare_dataset.csv" 
target_column = 'diabetes'  
numerical_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_columns = ['gender', 'smoking_history']

#Loading data
X_train, X_test, X_val, y_train, y_test, y_val = load_data(file_path, target_column)

#Applying feature engineering
X_train = create_features(X_train, numerical_columns, categorical_columns)
X_test = create_features(X_test, numerical_columns, categorical_columns)
X_val = create_features(X_val, numerical_columns, categorical_columns)

#Training models
logistic_model = train_logistic_model(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

#Evaluating models on test set
print("Evaluation on Test Set:")
logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
rf_metrics = evaluate_model(rf_model, X_test, y_test)

print("Logistic Regression Test Metrics:", logistic_metrics)
print("Random Forest Test Metrics:", rf_metrics)

#Evaluating models on validation set
print("\nEvaluation on Validation Set:")
logistic_val_metrics = evaluate_model(logistic_model, X_val, y_val)
rf_val_metrics = evaluate_model(rf_model, X_val, y_val)

print("Logistic Regression Validation Metrics:", logistic_val_metrics)
print("Random Forest Validation Metrics:", rf_val_metrics)
