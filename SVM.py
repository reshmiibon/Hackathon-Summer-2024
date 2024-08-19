#libraries to import 
import pandas as pd #for data manipulation and analysis in Python
import numpy as np #used for numerical computations in Python.
from sklearn.svm import SVC #import support vector classifier, machine learning model 
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report #imports multiple evaluation metrics used to assess the performance of classification models
from sklearn.model_selection import train_test_split, RandomizedSearchCV  #RandomizedSearchCV - used for hyperparameter tuning 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures #data preprocessing classes  


def load_data():
    # Read data from a tab-delimited text files and store it in a DataFrame
    meta_data = pd.read_csv('data/Tab_delimited_text/Hackathon2024.Meta.txt.gz', sep='\t')
    rna_data = pd.read_csv('data/Tab_delimited_text/Hackathon2024.RNA.txt.gz', sep='\t', index_col=0) #uses the first column as the index
    atac_data = pd.read_csv('data/Tab_delimited_text/Hackathon2024.ATAC.txt.gz', sep='\t', index_col=0)
    train_pairs = pd.read_csv('data/Tab_delimited_text/Hackathon2024.Training.Set.Peak2Gene.Pairs.txt.gz', sep='\t')
    test_pairs = pd.read_csv('data/Tab_delimited_text/Hackathon2024.Testing.Set.Peak2Gene.Pairs.txt.gz', sep='\t')
    
    return meta_data, rna_data, atac_data, train_pairs, test_pairs


def preprocess_data(data_pairs, rna_data, atac_data):
    #creating new features (columns) to add meaning to data 

    #adding RNA mean feature -- the mean (average) of the expression levels of a particular gene across all the samples.
    data_pairs['rna_mean'] = data_pairs.apply( #adding a rna_mean column to data_pairs dataframe 
        lambda x: np.mean(  #apply the mean function to x 
        rna_data.loc[x['gene'], :]) #selects the row in rna_data corresponding to the gene specified in the current row of data_pair 
        if x['gene'] in rna_data.index else 0, #(only if the gene exists in rna_data)
        axis=1) #the mean is calculated ROW WISE of the data_pairs dataframe
    
    #adding RNA median feature -- the median of the expression levels of a particular gene across all the samples.
    data_pairs['rna_median'] = data_pairs.apply(
        lambda x: np.median(rna_data.loc[x['gene'], :]) if x['gene'] in rna_data.index else 0, axis=1
    )

    #adding RNA std feature -- the standard deviation of the expression levels of a particular gene across all the samples.
    data_pairs['rna_std'] = data_pairs.apply(
        lambda x: np.std(rna_data.loc[x['gene'], :]) if x['gene'] in rna_data.index else 0, axis=1
    )

    #adding RNA nonzero feature -- the count of non-zero expression levels of a particular gene across all the samples.
    data_pairs['rna_nonzero'] = data_pairs.apply(
        lambda x: np.count_nonzero(rna_data.loc[x['gene'], :])  if x['gene'] in rna_data.index else 0, axis=1
    )
    
    #adding ATAC mean feature --  the average (mean) accessibility level of a specific peak across all samples
    data_pairs['atac_mean'] = data_pairs.apply(
        lambda x: np.mean(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )
    
    #adding ATAC median feature -- the median accessibility level of a specific peak across all samples
    data_pairs['atac_median'] = data_pairs.apply(
        lambda x: np.median(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )

    #adding ATAC std feature -- the standard deviation of the accessibility level of a specific peak across all samples
    data_pairs['atac_std'] = data_pairs.apply(
        lambda x: np.std(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )

    #adding ATAC nonzero feature -- the count of non-zero accessibility level of a specific peak across all samples
    data_pairs['atac_nonzero'] = data_pairs.apply(
        lambda x: np.count_nonzero(atac_data.loc[x['peak'], :]) if x['peak'] in atac_data.index else 0, axis=1
    )
    
    #adding mean product feature -- the product of rna_mean and atac_mean 
    data_pairs['mean_product'] = data_pairs['rna_mean'] * data_pairs['atac_mean']
    
    #feature scaling to have a mean of 0 and standard deviation of 1 helps improve the performance and stability of model 
    features = ['rna_mean', 'rna_median', 'rna_std', 'rna_nonzero',
                'atac_mean', 'atac_median', 'atac_std', 'atac_nonzero', 'mean_product']
    
    scaler = StandardScaler()
    data_pairs[features] = scaler.fit_transform(data_pairs[features])
    
    #generates new features based on polynomial combinations of the existing features
    poly = PolynomialFeatures(degree=2, #generates new features by taking combinations of the input features up to degree 2 (i.e x1, x2, x1*x2)
                              interaction_only=True, #generates only interaction terms between features, excluding squared terms (i.e x1^2, x2^2)
                              include_bias=False) #does not include a bias term (constant term of 1) in output since data is already centered (mean = 0) after scaling
    poly_features = poly.fit_transform(data_pairs[features]) #"fits" the transformer to the data and then transforms the input data into a new feature set
    poly_feature_names = poly.get_feature_names_out(features) #generates a list of names for the polynomial features
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data_pairs.index) #converts the NumPy array of polynomial features into a Pandas DataFrame
    
    # Concatenate with the original data (excluding the original features if necessary)
    data_pairs = pd.concat([data_pairs.drop(columns=features), poly_df], axis=1)
    
    return data_pairs

def train_model(X, y):
    # Define a parameter grid for hyperparameter tuning with SVM
    # potential hyperparameters for the SVM model that will be tuned to find the best combination.
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Initialize the RandomizedSearchCV object for SVM
    random_search = RandomizedSearchCV( 
        estimator=SVC(), #model is Support Vector Classifier (SVC) 
        param_distributions=param_grid, #the hyperparameter grid 
        n_iter=20, #20 different combinations of the hyperparameters will be sampled from the grid and evaluated
        cv=3, #3-fold cross-validation used to evaluate each combination of hyperparameters -- data will be split into 3 subsets, and the model will be trained on 2 subsets and tested on the remaining one, repeating this process 3 times with different subsets each time.
        scoring='accuracy', #model's performance will be evaluated based on accuracy during cross-validation
        n_jobs=-1, verbose=1, random_state=42
    )
    
    # randomized search will evaluate different combinations of hyperparameters
    random_search.fit(X, y) 
    
    # retrieves the best model (highest accuracy)
    best_model = random_search.best_estimator_
    
    return best_model

# making predictions on test data  
def predict_test_data(model, test_pairs, rna_data, atac_data):
    test_pairs = preprocess_data(test_pairs, rna_data, atac_data)
    X_test = test_pairs.drop(columns=['peak', 'gene', 'Pair', 'Peak2Gene'], errors='ignore')
    test_pairs['Peak2Gene'] = model.predict(X_test)
    return test_pairs

# save predictions to prediction file 
def save_predictions(test_pairs):
    test_pairs['Peak2Gene'] = test_pairs['Peak2Gene'].apply(lambda x: 'TRUE' if x==1 else 'FALSE')
    test_pairs[['peak', 'gene', 'Pair', 'Peak2Gene']].to_csv('prediction/prediction.csv', index=False)

# Main workflow
meta_data, rna_data, atac_data, train_pairs, test_pairs = load_data()

# After preprocessing, inspect first few rows and summary statistics
train_pairs = preprocess_data(train_pairs, rna_data, atac_data)

# Prepare training data
y_train_full = train_pairs['Peak2Gene'].apply(lambda x: 1 if x else 0)
X_train_full = train_pairs.drop(columns=['Peak2Gene', 'peak', 'gene', 'Pair'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# Train the model on the training set with hyperparameter tuning using RandomizedSearchCV
model = train_model(X_train, y_train)

# Evaluate on validation data
y_val_pred = model.predict(X_val)
print("\nValidation Data Evaluation with Tuned Model:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"MCC: {matthews_corrcoef(y_val, y_val_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Train the final model on the entire training data
final_model = train_model(X_train_full, y_train_full)

# Predict on the test data using the final model
test_pairs = predict_test_data(final_model, test_pairs, rna_data, atac_data)

# Save the predictions
save_predictions(test_pairs)
