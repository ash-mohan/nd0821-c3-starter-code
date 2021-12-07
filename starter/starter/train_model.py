# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, export_model_files
import pandas as pd
import pickle

# Add code to load in the data.
data = pd.read_csv("../data/census_data_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# get training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

print("Beginning Training Process...")
# Train and save a model.
model = train_model(X_train, y_train)
y_train_preds = model.predict(X_train)
y_test_preds = model.predict(X_test)

precision, recall, fbeta, accuracy = compute_model_metrics(y_train, y_train_preds)
print(f"\nTraining Accuracy: {accuracy}")
print(f"Training F-beta score: {fbeta}")
print(f"Training Precision: {precision}")
print(f"Training Recall: {recall}")

precision, recall, fbeta, accuracy = compute_model_metrics(y_test, y_test_preds)
print(f"\nTesting Accuracy: {accuracy}")
print(f"Testing F-beta score: {fbeta}")
print(f"Testing Precision: {precision}")
print(f"Testing Recall: {recall}")


# Train on entire dataset , this model will be used for inference
X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)

final_model = train_model(X, y)
y_pred = model.predict(X)
precision, recall, fbeta, accuracy = compute_model_metrics(y, y_pred)
print(f"\nFinal Model Accuracy: {accuracy}")
print(f"Final Model F-beta score: {fbeta}")
print(f"Final Model Precision: {precision}")
print(f"Final Model Recall: {recall}")

# save model and other files
export_model_files(final_model, encoder, lb)