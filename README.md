# System Threat Forecaster

A machine learning project for predicting system security threats using Windows Defender telemetry data. This project was developed as part of a Kaggle competition.

## 📊 Project Overview

This project implements a machine learning solution to predict potential security threats on computer systems using various system characteristics and configurations. The model analyzes features like antivirus configurations, hardware specifications, OS details, and security settings to classify systems as potentially infected or clean.

## 🎯 Objective

Develop a robust classification model that can:
- Predict potential system security threats
- Identify key factors that contribute to system vulnerability
- Provide actionable insights for system security improvement

## 📈 Results

- **Model**: LightGBM Classifier
- **Validation Accuracy**: 63.0%
- **Key Performance Metrics**:
  - Precision (Not Infected): 63.7%
  - Precision (Infected): 62.4%
  - Recall (Not Infected): 58.2%
  - Recall (Infected): 67.7%

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
