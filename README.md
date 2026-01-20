# Student Performance Prediction

A machine learning project that predicts student math scores based on various demographic and academic factors using multiple regression models.

## Overview

This project analyzes student performance data and builds predictive models to estimate math scores based on factors like gender, parental education, test preparation, and other academic scores. The project compares five different machine learning algorithms to identify the best performing model.

## Dataset

The dataset contains student performance information with the following features:

- **Gender**: Student's gender
- **Race/Ethnicity**: Student's ethnic group (Groups A-E)
- **Parental Level of Education**: Parents' highest education level
- **Lunch**: Type of lunch (standard or free/reduced)
- **Test Preparation Course**: Whether student completed test prep
- **Reading Score**: Score in reading (0-100)
- **Writing Score**: Score in writing (0-100)
- **Math Score**: Score in math (0-100) - *Target Variable*

Data source: [StudentsPerformance.csv](https://raw.githubusercontent.com/Prithivi-12/student_performance_prediction/main/StudentsPerformance.csv)

## Project Structure

```
├── Student_Performance_Prediction-1.ipynb  # Main notebook
├── README.md                                # Project documentation
├── best_model.pkl                           # Saved best performing model
├── scaler.pkl                               # Saved feature scaler
├── label_encoders.pkl                       # Saved label encoders
├── model_comparison_results.csv             # Model performance comparison
└── visualizations/
    ├── correlation_heatmap.png
    ├── score_distributions.png
    ├── outlier_boxplot.png
    └── model_comparison.png
```

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - matplotlib & seaborn - Data visualization
  - scikit-learn - Machine learning models and preprocessing
  - ipywidgets - Interactive widgets for Jupyter
  - pickle - Model serialization

## Machine Learning Models

The project implements and compares five regression algorithms:

1. **Linear Regression** - Basic linear approach
2. **Ridge Regression** - Regularized linear regression
3. **Decision Tree Regressor** - Non-linear tree-based model
4. **Random Forest Regressor** - Ensemble of decision trees
5. **Gradient Boosting Regressor** - Advanced boosting ensemble method

## Workflow

### 1. Data Loading & Preprocessing
- Load dataset from GitHub repository
- Check for missing values and duplicates
- Encode categorical variables using Label Encoding
- Create additional features (average score)

### 2. Exploratory Data Analysis (EDA)
- Generate correlation heatmap
- Visualize score distributions
- Detect outliers using boxplots
- Analyze relationships between features

### 3. Feature Engineering
- Select relevant features for prediction
- Create derived features (average score)
- Split data into training (80%) and testing (20%) sets
- Apply StandardScaler for feature normalization

### 4. Model Training
- Train all five models on scaled training data
- Use consistent random states for reproducibility

### 5. Model Evaluation
- Evaluate models using:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R² Score** (Coefficient of Determination)
- Compare model performance visually
- Select best performing model based on R² score

### 6. Prediction
- Create prediction function for new data
- Demonstrate predictions with sample examples
- Save trained models for deployment

### 7. Model Persistence
- Save best model using pickle
- Save preprocessing objects (scaler, encoders)
- Export model comparison results

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

### Running the Notebook
1. Clone the repository
2. Open `Student_Performance_Prediction-1.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially

### Making Predictions

```python
import pickle
import numpy as np

# Load saved models
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prepare input data
# Format: [gender, race/ethnicity, parental_education, lunch, test_prep, reading_score, writing_score]
input_data = np.array([[1, 2, 3, 0, 1, 75, 78]])

# Scale and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

print(f"Predicted Math Score: {prediction[0]:.2f}")
```

### Encoding Reference

**Gender**: 0 = Female, 1 = Male

**Race/Ethnicity**: 0-4 (Groups A-E)

**Parental Education**: 
- 0 = Some high school
- 1 = High school
- 2 = Some college
- 3 = Associate's degree
- 4 = Bachelor's degree
- 5 = Master's degree

**Lunch**: 0 = Standard, 1 = Free/Reduced

**Test Preparation**: 0 = None, 1 = Completed

## Results

The project successfully builds and compares multiple models to predict student math performance. Model comparison results are saved in `model_comparison_results.csv` with detailed metrics for each algorithm.

## Key Insights

- Strong correlation between reading, writing, and math scores
- Test preparation course completion positively impacts performance
- Parental education level shows influence on student scores
- Ensemble methods (Random Forest, Gradient Boosting) generally perform better than simpler models

## Future Enhancements

- Implement hyperparameter tuning for better performance
- Add cross-validation for robust evaluation
- Create web-based deployment using Streamlit or Flask
- Explore deep learning approaches
- Add feature importance analysis
- Implement prediction confidence intervals

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Author

Developed as a machine learning demonstration project for student performance analysis.

## Acknowledgments

- Dataset source: Student Performance Dataset
- Inspired by educational data mining research
