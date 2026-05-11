# Road Accident Severity Prediction System

A Machine Learning project that predicts accident severity using road accident data and classification algorithms such as Support Vector Machine and Logistic Regression.

The system analyzes accident-related factors including driver information, road conditions, weather conditions, collision types, and vehicle details to predict whether an accident will result in slight injury or severe injury.

## Project Phases
### Phase 1 — Data Cleaning
#### Objectives
- Handle missing values
- Remove inconsistent categories
- Reduce category complexity
- Prepare dataset for analysis

#### Tasks Performed
- Removed unnecessary columns
- Handled null values
- Standardized categorical values
- Reduced high-cardinality categories
- Created cleaned dataset

#### Output
- `cleaned_dataset.csv`



### Phase 2 — EDA, Preprocessing & Model Training
#### Exploratory Data Analysis (EDA)

Performed data visualization and statistical analysis using:

- Class distribution plots
- Correlation heatmaps
- Count plots
- Boxplots
- Confusion matrices

#### Preprocessing
- One-hot encoding
- Feature selection using SelectKBest
- Feature scaling using StandardScaler
- Class balancing using SMOTE
- Train-test split

#### Models Used
- SVM Linear Kernel
- SVM RBF Kernel
- SVM Polynomial Kernel
- SVM Sigmoid Kernel
- Logistic Regression

#### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

#### Final Model Selection

The Polynomial Kernel SVM achieved the best balance between:

- severe accident detection
- macro F1-score
- overall classification balance

### Phase 3 — Implementation
#### Features
- Load trained models
- Load preprocessing artifacts
- Predict accident severity
- Test real-world accident scenarios

#### Saved Models
```
svm_linear_model.pkl
svm_rbf_model.pkl
svm_poly_model.pkl
svm_sigmoid_model.pkl
logistic_model.pkl
```

#### Saved Preprocessing Files
```
scaler.pkl
selected_features.pkl
target_encoder.pkl
```

#### Prediction Output

The system predicts:

- Slight Injury
- Severe Injury

along with prediction confidence scores.

```
Project Structure
├── cleaning.ipynb
├── preprocessing.ipynb
├── training.ipynb
├── implement.ipynb
├── cleaned_dataset.csv
├── scaler.pkl
├── selected_features.pkl
├── target_encoder.pkl
├── svm_linear_model.pkl
├── svm_rbf_model.pkl
├── svm_poly_model.pkl
├── svm_sigmoid_model.pkl
├── logistic_model.pkl
└── README.md
```

## Technologies Used
- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- joblib

#### Conclusion

The project successfully demonstrates how machine learning can be used to predict accident severity using road accident data. Among all tested models, the Polynomial Kernel SVM provided the best balance between severe accident detection and overall model performance.

Future improvements can include:

- Real-time traffic integration
- Larger datasets
- Deep learning approaches
- Web/mobile deployment
- Live accident risk prediction