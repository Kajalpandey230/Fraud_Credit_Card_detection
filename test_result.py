import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the test CSV
test_data = pd.read_csv('fraudTest.csv')

# Preprocessing steps like filling null values, feature extraction, encoding, etc.
test_data.fillna(0, inplace=True)  # Handle missing values as per your approach

# Extract features from datetime columns (as per the original training data)
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])
test_data['dob'] = pd.to_datetime(test_data['dob'])

# Example of extracting features like hour, day, month, etc.
test_data['trans_hour'] = test_data['trans_date_trans_time'].dt.hour
test_data['trans_day'] = test_data['trans_date_trans_time'].dt.day
test_data['trans_month'] = test_data['trans_date_trans_time'].dt.month

# Drop the original datetime column after extraction
test_data.drop('trans_date_trans_time', axis=1, inplace=True)

# Convert 'dob' into age (optional: based on current date)
today = pd.to_datetime('today')
test_data['age'] = (today - test_data['dob']).dt.days // 365
test_data.drop('dob', axis=1, inplace=True)

# Drop any irrelevant columns (update this based on your actual dataset and model)
test_data.drop(['first', 'last', 'street', 'trans_num', 'Unnamed: 0'], axis=1, inplace=True)

# Handle categorical features - you should know which columns were encoded during training
categorical_columns = ['merchant', 'category', 'gender', 'state', 'job', 'city']

# Load the encoder for categorical columns (if used during training)
encoder = LabelEncoder()

for col in categorical_columns:
    test_data[col] = encoder.fit_transform(test_data[col].astype(str))

# Now, ensure that the test data has the same columns as the training data.
# Here, load your training data to get the exact column names.
# For simplicity, we're assuming you know which features were used during training.

# Select only the columns used for training
# Assuming the model was trained on specific columns like ['amt', 'trans_hour', 'trans_day', 'trans_month', 'age', ...]
# Match them accordingly.
train_columns = ['amt', 'trans_hour', 'trans_day', 'trans_month', 'age', 'merchant', 'category', 'gender', 'city']

# Filter the test data to only include the training columns
X_test = test_data[train_columns]

# Load the trained model (ensure the correct path and filename)
model = joblib.load('fraud_detection_model.jb')  # Update the file path if needed

# Separate the true labels (y_true)
y_true = test_data['is_fraud']

# Make predictions using the model
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
