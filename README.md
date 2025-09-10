# System Threat Forecaster

🛡️ ML-based system threat detection with 5 optimized models achieving 60-63% accuracy through advanced hyperparameter tuning

A machine learning project for predicting system security threats using Windows Defender telemetry data. This project was developed as part of a Kaggle competition.

### � Technical Achievements

#### Hyperparameter Tuning Successes
- **LightGBM**: Optimized with RandomizedSearchCV (100 iterations)
  - Best parameters: n_estimators=200, learning_rate=0.1, num_leaves=31
- **XGBoost**: Advanced configuration with early stopping
  - Optimal parameters: n_estimators=200, learning_rate=0.1, max_depth=6
- **Random Forest**: Grid search optimization with memory management
  - Best parameters: n_estimators=200, max_depth=10, min_samples_split=2
- **SGD Classifier**: Comprehensive hyperparameter exploration
  - Optimal parameters: alpha=0.01, learning_rate='constant', loss='hinge'

#### Memory Optimization & Data Processing
- Implemented sparse matrix preprocessing to handle large datasets efficiently
- Optimized hyperparameter tuning with memory-conscious parameter grids
- Consistent preprocessing pipeline across all models for fair comparison
- Error handling and fallback strategies for robust model training

### 🚀 Key Learnings

1. **XGBoost with Advanced Features**: Early stopping and optimal hyperparameters achieved the best performance at 63.1%
2. **LightGBM Efficiency**: Excellent balance of speed and accuracy, making it highly practical for deployment
3. **Ensemble Methods**: Random Forest provided robust performance with good interpretability
4. **Linear Models**: SGD classifier proved surprisingly effective for this dataset type
5. **Neural Networks**: MLP achieved competitive results, showing the dataset's suitability for deep learning approaches

## 📊 Model Insights

This project implements a machine learning solution to predict potential security threats on computer systems using various system characteristics and configurations. The model analyzes features like antivirus configurations, hardware specifications, OS details, and security settings to classify systems as potentially infected or clean.

## 🎯 Objective

Develop a robust classification model that can:
- Predict potential system security threats
- Identify key factors that contribute to system vulnerability
- Provide actionable insights for system security improvement

## 📈 Results

### 🏆 Model Performance Comparison

| Model | Validation Accuracy | Cross-Validation Accuracy | Key Strengths |
|-------|-------------------|---------------------------|---------------|
| **XGBoost (Advanced)** | **63.1%** | - | Best overall performance, early stopping |
| **LightGBM (Tuned)** | **62.9%** | **60.5%** | Fast training, optimal feature handling |
| **Random Forest (Tuned)** | **60.9%** | **60.5%** | Robust ensemble method |
| **SGD Classifier (Tuned)** | **60.9%** | **60.7%** | Memory efficient, linear model |
| **MLP Classifier** | **61.9%** | - | Neural network approach |

### 🎯 Best Model Performance (LightGBM)
- **Validation Accuracy**: 62.9%
- **Key Performance Metrics**:
  - Precision (Not Infected): 63.6%
  - Precision (Infected): 62.3%
  - Recall (Not Infected): 58.1%
  - Recall (Infected): 67.6%
  - F1-Score (Not Infected): 60.7%
  - F1-Score (Infected): 64.8%

## 🔍 Key Features Identified

The model identified the following as the most important features for threat prediction:

1. **NumAntivirusProductsInstalled** (15.0% importance)
2. **AntivirusConfigID** (11.8% importance)
3. **IsAlwaysOnAlwaysConnectedCapable** (6.9% importance)
4. **TotalPhysicalRAMMB** (6.7% importance)
5. **IsSystemProtected** (6.2% importance)

## 📁 Project Structure

```
System-Threat-Forecaster/
│
├── 21f2001203-notebook-t12025 (1).ipynb    # Main analysis notebook
├── train.csv                               # Training dataset
├── test.csv                                # Test dataset
├── sample_submission.csv                   # Sample submission format
├── submission.csv                          # Final predictions
└── README.md                              # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.11**
- **Libraries**:
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - scikit-learn - Machine learning algorithms
  - LightGBM - Gradient boosting framework
  - matplotlib & seaborn - Data visualization
  - plotly - Interactive visualizations

## 📊 Dataset Information

- **Training Data**: 100,000 samples with 76 features
- **Test Data**: 10,000 samples with 75 features
- **Target Variable**: Binary classification (0: Not Infected, 1: Infected)
- **Features Include**:
  - System hardware specifications
  - Operating system details
  - Antivirus configurations
  - Security settings
  - Geographic information
  - User behavior patterns

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn plotly
```

### Running the Project

1. Clone this repository
2. Ensure all CSV files are in the same directory as the notebook
3. Open and run `21f2001203-notebook-t12025 (1).ipynb` in Jupyter Notebook or VS Code
4. The model will train and generate predictions in `submission.csv`

## 📋 Methodology

1. **Exploratory Data Analysis (EDA)**
   - Data distribution analysis
   - Feature correlation analysis
   - Missing value assessment

2. **Data Preprocessing**
   - Train-validation split (80-20)
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features
   - Feature engineering resulting in 3,394 final features

3. **Model Development**
   - LightGBM Classifier implementation
   - Hyperparameter optimization
   - Cross-validation for model evaluation

4. **Model Evaluation**
   - Confusion matrix analysis
   - Classification report
   - Feature importance analysis

## 📊 Model Performance

The LightGBM model achieved a validation accuracy of **63.0%**, which is quite good for this type of security classification task considering:
- The complexity of system threat detection
- The balanced nature of the dataset
- The high-dimensional feature space

## 🔮 Future Improvements

- **Feature Engineering**: Create additional domain-specific features
- **Model Ensemble**: Combine multiple algorithms for better performance
- **Hyperparameter Tuning**: More extensive grid search optimization
- **Feature Selection**: Apply advanced feature selection techniques
- **Deep Learning**: Experiment with neural network architectures

## 📄 License

This project is part of academic coursework for IIT Madras BS in Data Science & Applications.

## 👨‍💻 Author

**Student ID**: 21f2001203  
**Course**: Machine Learning Practice  
**Institution**: IIT Madras BS in Data Science & Applications

---

*This project demonstrates the application of machine learning techniques to cybersecurity challenges, specifically in predicting system vulnerabilities based on configuration and usage patterns.*
