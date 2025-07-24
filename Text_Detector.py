import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import torch
from transformers import BertTokenizer, BertModel
import joblib
import os

# Initialize variables

filename = "balanced_dataset_15000.csv"

# Function to load dataset
def loadDataset():
    dataset = pd.read_csv(filename)
    labels = ['Human', 'AI']
    return dataset, labels

# Function to extract features using BERT
def featuresExtraction(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_features(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    bert_features = np.array([get_bert_features(text) for text in data])
    print("Number of features extracted by BERT:", bert_features.shape[1])
    return bert_features

# Function to apply PCA and scale the dataset
def PCASelection(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    print("Number of features remaining after PCA:", X.shape[1])
    return X, scaler, pca

# Function to train the XGBoost model
def trainXGBoost(X_train, y_train):
    xgb_cls = XGBClassifier()
    xgb_cls.fit(X_train, y_train)
    return xgb_cls

# Function to save models, components, and BERT features
def saveComponents(xgb_cls, scaler, pca, bert_features, X_test, y_test):
    joblib.dump(xgb_cls, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    np.save('bert_features.npy', bert_features)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

# Function to load models, components, and BERT features
def loadComponents():
    xgb_cls = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    bert_features = np.load('bert_features.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    return xgb_cls, scaler, pca, bert_features, X_test, y_test

# Function to predict text using the trained model
def predictText(text_data, xgb_cls, scaler, pca):
    data = text_data
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    def get_bert_features(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    bert_feature = get_bert_features(data)
    bert_feature = bert_feature.reshape(1, -1)
    bert_feature = scaler.transform(bert_feature)
    bert_feature = pca.transform(bert_feature)

    predict = xgb_cls.predict(bert_feature)
    predict = predict[0]
    if predict == 0:
        return "Predicted Text is: Human Written"
    else:
        return "Predicted Text is: AI Generated"

# Main script
def main():
    if os.path.exists('xgb_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('pca.pkl'):
        xgb_cls, scaler, pca, bert_features, X_test, y_test = loadComponents()
        X_test_pca = X_test
        print("Number of features extracted by BERT:", bert_features.shape[1])
        
        # Print the number of features remaining after PCA
        print("Number of features remaining after PCA:", X_test_pca.shape[1])
        
    else:
        dataset, labels = loadDataset()
        data = dataset['text'].values
        target = dataset['generated'].values

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
        
        # Feature extraction
        X_train_bert = featuresExtraction(X_train)
        X_test_bert = featuresExtraction(X_test)
        
        # Apply PCA
        X_train_pca, scaler, pca = PCASelection(X_train_bert)
        X_test_pca = scaler.transform(X_test_bert)
        X_test_pca = pca.transform(X_test_pca)

        # Train the model
        xgb_cls = trainXGBoost(X_train_pca, y_train)

        # Save components
        saveComponents(xgb_cls, scaler, pca, X_train_bert, X_test_pca, y_test)

    text_data = input("Enter your text here: ")
    prediction = predictText(text_data, xgb_cls, scaler, pca)
    print(prediction)

    # Evaluate model performance
    y_pred = xgb_cls.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

main()

