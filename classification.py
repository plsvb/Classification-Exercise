import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load data
train_url = 'https://github.com/WHPAN0108/BHT-DataScience-S23/blob/main/classification/data/Assigment/aug_train.csv?raw=true'
test_url = 'https://github.com/WHPAN0108/BHT-DataScience-S23/blob/main/classification/data/Assigment/aug_test.csv?raw=true'

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# Data Cleaning
def clean_data(df):
    df['experience'] = df['experience'].replace({'>20': '21', '<1': '1'}).astype(float)
    df['last_new_job'] = df['last_new_job'].replace({'>4': '5', 'never': '0'}).astype(float)
    
    # Impute missing values
    for column in df.columns:
        if df[column].dtype == 'object':
            mode = df[column].mode()[0]
            df[column] = df[column].fillna(mode)
        else:
            median = df[column].median()
            df[column] = df[column].fillna(median)
    
    return df

train_df = clean_data(train_df)
test_df = clean_data(test_df)

# Encode categorical variables
label_encoders = {}
for column in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[column] = le.fit_transform(train_df[column])
    test_df[column] = le.transform(test_df[column])
    label_encoders[column] = le

# Separate features and target
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']
# Build a classification model (Random Forest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions on training set
y_train_pred = clf.predict(X_train)

# Evaluation on training set
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Predictions on test set
y_test_pred = clf.predict(X_test)

# Evaluation on test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print results
print("Training Set Evaluation:")
print("Confusion Matrix:\n", train_conf_matrix)
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-Score:", train_f1)

print("\nTest Set Evaluation:")
print("Confusion Matrix:\n", test_conf_matrix)
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-Score:", test_f1)
