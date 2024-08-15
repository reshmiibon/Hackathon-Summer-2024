import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report

# Load the data
def load_data():
    meta_data = pd.read_csv('data/Tab_delimited_text/Hackathon2024.Meta.txt.gz', sep='\t')
    rna_data = pd.read_csv('data/Tab_delimited_text/Hackathon2024.RNA.txt.gz', sep='\t', index_col=0)
    atac_data = pd.read_csv('data/Tab_delimited_text/Hackathon2024.ATAC.txt.gz', sep='\t', index_col=0)
    train_pairs = pd.read_csv('data/Tab_delimited_text/Hackathon2024.Training.Set.Peak2Gene.Pairs.txt.gz', sep='\t')
    test_pairs = pd.read_csv('data/Tab_delimited_text/Hackathon2024.Testing.Set.Peak2Gene.Pairs.txt.gz', sep='\t')
    
    return meta_data, rna_data, atac_data, train_pairs, test_pairs

# Preprocess the data
def preprocess_data(data_pairs, rna_data, atac_data):
    # Calculate statistical features
    data_pairs['rna_mean'] = data_pairs.apply(
        lambda x: np.mean(rna_data.loc[x['gene'], :]) if x['gene'] in rna_data.index else 0, axis=1
    )
    data_pairs['rna_median'] = data_pairs.apply(
        lambda x: np.median(rna_data.loc[x['gene'], :]) if x['gene'] in rna_data.index else 0, axis=1
    )
    data_pairs['rna_std'] = data_pairs.apply(
        lambda x: np.std(rna_data.loc[x['gene'], :]) if x['gene'] in rna_data.index else 0, axis=1
    )
    data_pairs['rna_nonzero'] = data_pairs.apply(
        lambda x: np.count_nonzero(rna_data.loc[x['gene'], :]) if x['gene'] in rna_data.index else 0, axis=1
    )
    
    data_pairs['atac_mean'] = data_pairs.apply(
        lambda x: np.mean(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )
    data_pairs['atac_median'] = data_pairs.apply(
        lambda x: np.median(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )
    data_pairs['atac_std'] = data_pairs.apply(
        lambda x: np.std(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )
    data_pairs['atac_nonzero'] = data_pairs.apply(
        lambda x: np.count_nonzero(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )
    
    # Add the product of rna_mean and atac_mean as a new feature
    data_pairs['mean_product'] = data_pairs['rna_mean'] * data_pairs['atac_mean']
    
    # Feature scaling
    features = ['rna_mean', 'rna_median', 'rna_std', 'rna_nonzero',
                'atac_mean', 'atac_median', 'atac_std', 'atac_nonzero', 'mean_product']
    
    scaler = StandardScaler()
    data_pairs[features] = scaler.fit_transform(data_pairs[features])
    
    # Generate interaction and polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data_pairs[features])
    poly_feature_names = poly.get_feature_names_out(features)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data_pairs.index)
    
    # Concatenate with the original data (excluding the original features if necessary)
    data_pairs = pd.concat([data_pairs.drop(columns=features), poly_df], axis=1)
    
    return data_pairs

# Train the XGBoost model
def train_xgboost_model(X, y):
    # Define a parameter grid for hyperparameter tuning with XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    grid_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=1,
        random_state=42
    )

    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    
    return best_model

# Predict on test data
def predict_test_data(model, test_pairs, rna_data, atac_data):
    test_pairs = preprocess_data(test_pairs, rna_data, atac_data)
    X_test = test_pairs.drop(columns=['peak', 'gene', 'Pair', 'Peak2Gene'], errors='ignore')
    test_pairs['Peak2Gene'] = model.predict(X_test)
    return test_pairs

# Save the predictions
def save_predictions(test_pairs):
    test_pairs['Peak2Gene'] = test_pairs['Peak2Gene'].apply(lambda x: 'TRUE' if x==1 else 'FALSE')
    test_pairs[['peak', 'gene', 'Pair', 'Peak2Gene']].to_csv('prediction/prediction.csv', index=False)

# Main workflow
meta_data, rna_data, atac_data, train_pairs, test_pairs = load_data()

# Preprocess the training data
train_pairs = preprocess_data(train_pairs, rna_data, atac_data)

# Prepare training data
y_train_full = train_pairs['Peak2Gene'].apply(lambda x: 1 if x else 0)
X_train_full = train_pairs.drop(columns=['Peak2Gene', 'peak', 'gene', 'Pair'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# Train the model on the training set with hyperparameter tuning using RandomizedSearchCV
model = train_xgboost_model(X_train, y_train)

# Evaluate on validation data
y_val_pred = model.predict(X_val)
print("\nValidation Data Evaluation with Tuned Model:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"MCC: {matthews_corrcoef(y_val, y_val_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Optional: Train the final model on the entire training data
final_model = train_xgboost_model(X_train_full, y_train_full)

# Predict on the test data using the final model
test_pairs = predict_test_data(final_model, test_pairs, rna_data, atac_data)

# Save the predictions
save_predictions(test_pairs)

