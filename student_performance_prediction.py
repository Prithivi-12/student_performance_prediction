# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Set visualization style
sns.set_style("whitegrid")

# 2. Load Dataset
# Load dataset
student_dataset = pd.read_csv("dataset.csv")

# Display first few rows
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(student_dataset.head())
print(f"\nDataset shape: {student_dataset.shape}")

# 3. Exploratory Data Analysis (EDA)

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# 3.1 Target Distribution Visualization
plt.figure(figsize=(15, 5))

# Histogram of marks
plt.subplot(1, 3, 1)
plt.hist(student_dataset['marks'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.title('Distribution of Marks', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Boxplot of marks
plt.subplot(1, 3, 2)
plt.boxplot(student_dataset['marks'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.7))
plt.ylabel('Marks')
plt.title('Boxplot of Marks', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Pass/Fail distribution
plt.subplot(1, 3, 3)
pass_fail_counts = student_dataset['pass_fail'].value_counts().sort_index()
plt.bar(['Fail', 'Pass'], pass_fail_counts.values, color=['#ff6b6b', '#4ecdc4'], 
        edgecolor='black', alpha=0.7)
plt.xlabel('Pass/Fail')
plt.ylabel('Count')
plt.title('Pass/Fail Distribution', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 01_target_distribution.png")

# 3.2 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = student_dataset.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_correlation_heatmap.png")

# Print top correlations with marks
print("\nTop correlations with marks:")
marks_corr = correlation['marks'].sort_values(ascending=False)
for feature, corr_value in marks_corr.items():
    if feature != 'marks':
        print(f"  {feature}: {corr_value:.4f}")

# 3.3 Key Feature Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Study Hours vs Marks
axes[0, 0].scatter(student_dataset['study_hours'], student_dataset['marks'], 
                   alpha=0.5, color='#3498db', s=30, edgecolor='black', linewidth=0.3)
axes[0, 0].set_xlabel('Study Hours')
axes[0, 0].set_ylabel('Marks')
axes[0, 0].set_title('Study Hours vs Marks', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Attendance vs Marks
axes[0, 1].scatter(student_dataset['attendance'], student_dataset['marks'], 
                   alpha=0.5, color='#2ecc71', s=30, edgecolor='black', linewidth=0.3)
axes[0, 1].set_xlabel('Attendance (%)')
axes[0, 1].set_ylabel('Marks')
axes[0, 1].set_title('Attendance vs Marks', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Previous Scores vs Marks
axes[1, 0].scatter(student_dataset['previous_scores'], student_dataset['marks'], 
                   alpha=0.5, color='#e74c3c', s=30, edgecolor='black', linewidth=0.3)
axes[1, 0].set_xlabel('Previous Scores')
axes[1, 0].set_ylabel('Marks')
axes[1, 0].set_title('Previous Scores vs Marks', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Sleep Hours vs Marks
axes[1, 1].scatter(student_dataset['sleep_hours'], student_dataset['marks'], 
                   alpha=0.5, color='#9b59b6', s=30, edgecolor='black', linewidth=0.3)
axes[1, 1].set_xlabel('Sleep Hours')
axes[1, 1].set_ylabel('Marks')
axes[1, 1].set_title('Sleep Hours vs Marks', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_scatter_plots.png")

# 3.4 Categorical Features Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extracurricular vs Marks
sns.boxplot(x='extracurricular', y='marks', data=student_dataset, 
            ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Extracurricular Activities vs Marks', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# Family Support vs Marks
sns.boxplot(x='family_support', y='marks', data=student_dataset, 
            ax=axes[0, 1], palette='Set3')
axes[0, 1].set_title('Family Support vs Marks', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Internet Access vs Marks
sns.boxplot(x='internet_access', y='marks', data=student_dataset, 
            ax=axes[1, 0], palette='pastel')
axes[1, 0].set_title('Internet Access vs Marks', fontweight='bold')
axes[1, 0].set_xlabel('Internet Access (0=No, 1=Yes)')
axes[1, 0].grid(axis='y', alpha=0.3)

# Tutoring vs Marks
sns.boxplot(x='tutoring', y='marks', data=student_dataset, 
            ax=axes[1, 1], palette='muted')
axes[1, 1].set_title('Tutoring vs Marks', fontweight='bold')
axes[1, 1].set_xlabel('Tutoring (0=No, 1=Yes)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('04_categorical_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_categorical_features.png")
print("\n✓ EDA completed successfully\n")

# 4. Data Cleaning (Outlier Removal using IQR)

print("="*60)
print("DATA CLEANING")
print("="*60)

# Calculate IQR for study_hours
first_quartile = student_dataset['study_hours'].quantile(0.25)
third_quartile = student_dataset['study_hours'].quantile(0.75)
inter_quartile_range = third_quartile - first_quartile

# Remove outliers using IQR method
student_dataset = student_dataset[
    (student_dataset['study_hours'] >= first_quartile - 1.5 * inter_quartile_range) &
    (student_dataset['study_hours'] <= third_quartile + 1.5 * inter_quartile_range)
]

print("\nData cleaned successfully")
print(f"Dataset shape after cleaning: {student_dataset.shape}\n")

# 5. Feature and Target Separation

# Input features (all except targets)
student_input_features = student_dataset.drop(columns=['marks', 'pass_fail'])

# Regression target
student_marks_target = student_dataset['marks']

# Classification target
student_pass_fail_target = student_dataset['pass_fail']

# 6. Feature Scaling

# Initialize scaler
feature_scaler = StandardScaler()

# Fit and transform features
scaled_student_features = feature_scaler.fit_transform(student_input_features)

print("Data cleaned and scaled successfully\n")

# 7. Feature Selection

# Selected important features based on domain knowledge
important_features = [
    'study_hours',
    'attendance',
    'previous_scores',
    'sleep_hours',
    'extracurricular',
    'family_support',
    'tutoring'
]

# Extract selected features from dataset
selected_student_features = student_dataset[important_features]

# Scale selected features
scaled_selected_features = feature_scaler.fit_transform(selected_student_features)

print("Selected features:")
print(important_features)
print()

# 8. Linear Regression – Marks Prediction

print("="*60)
print("MODEL TRAINING")
print("="*60)

# Train-test split for regression (80-20 split)
train_features_regression, test_features_regression, train_marks, test_marks = train_test_split(
    scaled_selected_features,
    student_marks_target,
    test_size=0.2,
    random_state=42
)

# Initialize and train Linear Regression model
marks_prediction_model = LinearRegression()
marks_prediction_model.fit(train_features_regression, train_marks)

print(f"\nLinear Regression model trained")
print(f"Training samples: {len(train_features_regression)}, Testing samples: {len(test_features_regression)}")

# 9. Logistic Regression – Pass/Fail Prediction

# Train-test split for classification (80-20 split)
train_features_classification, test_features_classification, train_pass_fail, test_pass_fail = train_test_split(
    scaled_selected_features,
    student_pass_fail_target,
    test_size=0.2,
    random_state=42
)

# Initialize and train Logistic Regression model
pass_fail_prediction_model = LogisticRegression(max_iter=1000)
pass_fail_prediction_model.fit(train_features_classification, train_pass_fail)

print(f"Logistic Regression model trained")
print(f"Training samples: {len(train_features_classification)}, Testing samples: {len(test_features_classification)}\n")

# 10. Model Evaluation

print("="*60)
print("MODEL EVALUATION")
print("="*60)

# --- Linear Regression Evaluation ---

# Make predictions on test set
predicted_marks_test = marks_prediction_model.predict(test_features_regression)

# Calculate evaluation metrics
mean_squared_error_value = mean_squared_error(test_marks, predicted_marks_test)
root_mean_squared_error = np.sqrt(mean_squared_error_value)
r2_score_value = r2_score(test_marks, predicted_marks_test)

# Display results
print("\nLinear Regression Evaluation:")
print(f"R² Score: {r2_score_value:.4f}")
print(f"RMSE: {root_mean_squared_error:.4f}")

# --- Logistic Regression Evaluation ---

# Make predictions on test set
predicted_pass_fail_test = pass_fail_prediction_model.predict(test_features_classification)

# Calculate accuracy
classification_accuracy = accuracy_score(test_pass_fail, predicted_pass_fail_test)

# Display results
print("\nLogistic Regression Evaluation:")
print(f"Accuracy: {classification_accuracy:.4f}\n")

# 11. Visualization - Model Performance

print("="*60)
print("MODEL VISUALIZATION")
print("="*60)

# --- Linear Regression: Actual vs Predicted & Residual Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted scatter plot
axes[0].scatter(test_marks, predicted_marks_test, alpha=0.6, color='#3498db', 
                s=50, edgecolor='black', linewidth=0.5)
axes[0].plot([test_marks.min(), test_marks.max()], 
             [test_marks.min(), test_marks.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Marks', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Marks', fontsize=12, fontweight='bold')
axes[0].set_title(f'Actual vs Predicted\nR² = {r2_score_value:.4f}', 
                  fontweight='bold', fontsize=13)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Residual plot
residuals = test_marks - predicted_marks_test
axes[1].scatter(predicted_marks_test, residuals, alpha=0.6, color='#e74c3c', 
                s=50, edgecolor='black', linewidth=0.5)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Marks', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
axes[1].set_title('Residual Plot', fontweight='bold', fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_regression_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 05_regression_evaluation.png")

# --- Logistic Regression: Confusion Matrix ---
cm = confusion_matrix(test_pass_fail, predicted_pass_fail_test)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
            annot_kws={'size': 16, 'weight': 'bold'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix\nAccuracy: {classification_accuracy:.4f}', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('06_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_confusion_matrix.png\n")

# 12. Final Prediction (Sample Student Input)

print("="*60)
print("SAMPLE PREDICTION")
print("="*60)

# Sample student data
# Format: [study_hours, attendance, previous_scores, sleep_hours, extracurricular, family_support, tutoring]
new_student_data = np.array([[5, 85, 75, 7, 1, 1, 0]])

print("\nNew Student Data:")
for idx, feature in enumerate(important_features):
    print(f"  {feature}: {new_student_data[0][idx]}")

# Scale input using trained scaler
scaled_new_student_data = feature_scaler.transform(new_student_data)

# Predict marks using Linear Regression
predicted_final_marks = marks_prediction_model.predict(scaled_new_student_data)[0]

# Predict pass/fail using Logistic Regression
predicted_pass_fail_status = pass_fail_prediction_model.predict(scaled_new_student_data)[0]

# Get probability scores
pass_fail_probability = pass_fail_prediction_model.predict_proba(scaled_new_student_data)[0]

# Display prediction results
print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)
print(f"Predicted Marks: {round(predicted_final_marks, 2)}")
print(f"Prediction: {'Pass' if predicted_pass_fail_status == 1 else 'Fail'}")
print(f"Pass Probability: {pass_fail_probability[1]*100:.2f}%")
print(f"Fail Probability: {pass_fail_probability[0]*100:.2f}%\n")

# 13. Save Models

import pickle

print("="*60)
print("SAVING MODELS")
print("="*60)

# Save Linear Regression model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(marks_prediction_model, f)
print("\n✓ Linear Regression model saved: linear_regression_model.pkl")

# Save Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(pass_fail_prediction_model, f)
print("✓ Logistic Regression model saved: logistic_regression_model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
print("✓ Scaler saved: scaler.pkl")

print("\n✓ All models saved successfully!\n")

# Final Summary
print("="*60)
print("PROJECT SUMMARY")
print("="*60)

print(f"\nDataset: {student_dataset.shape[0]} samples, {len(important_features)} features")
print(f"\nModel Performance:")
print(f"  • Linear Regression - R² Score: {r2_score_value:.4f}, RMSE: {root_mean_squared_error:.4f}")
print(f"  • Logistic Regression - Accuracy: {classification_accuracy:.4f}")
print(f"\nGenerated Files:")
print("  Visualizations: 6 PNG files")
print("  Models: 3 PKL files")
print("\n" + "="*60)
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
