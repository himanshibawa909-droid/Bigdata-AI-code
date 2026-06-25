"""
Student Academic Performance Prediction using Machine Learning
MSc Data Science - Advanced Computing Project (COM7014)
Author: Himanshi Bawa
Date: 2024

This script implements machine learning algorithms for predicting student academic performance
using UCI Student Performance Dataset and Open University Learning Analytics Dataset (OULAD).

Algorithms: Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting (XGBoost)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class StudentPerformancePredictor:
    """
    Main class for student performance prediction pipeline.
    Handles data preprocessing, model training, and evaluation.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.results = {}
        self.models = {}
        
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return None
    
    def preprocess_data(self, df, target_column):
        """
        Preprocess the data:
        - Handle missing values
        - Encode categorical variables
        - Normalize numerical features
        - Create target variable (at-risk classification)
        """
        print("\n=== DATA PREPROCESSING ===")
        
        df_clean = df.copy()
        
        # Handle missing values
        print(f"Missing values before handling: {df_clean.isnull().sum().sum()}")
        
        # Fill missing values with mean for numerical, mode for categorical
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"Missing values after handling: {df_clean.isnull().sum().sum()}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        X = df_clean.drop(target_column, axis=1)
        y = df_clean[target_column]
        
        # Create classification target: at-risk (0), borderline (1), on-track (2)
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        else:
            # Convert continuous grades to classes
            y = pd.cut(y, bins=3, labels=[0, 1, 2])
            y = y.astype(int)
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:\n{pd.Series(y).value_counts().sort_index()}")
        
        return X, y
    
    def feature_selection(self, X, y, top_n=8):
        """
        Select most important features using correlation analysis and Random Forest.
        """
        print("\n=== FEATURE SELECTION ===")
        
        # Quick RF to get feature importance
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf_temp.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} features:")
        print(feature_importance.head(top_n))
        
        # Select top features
        selected_features = feature_importance.head(top_n)['feature'].tolist()
        X_selected = X[selected_features]
        
        return X_selected, selected_features, feature_importance
    
    def apply_smote(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        print("\n=== HANDLING CLASS IMBALANCE WITH SMOTE ===")
        
        print(f"Before SMOTE - Class distribution:\n{pd.Series(y).value_counts().sort_index()}")
        
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE - Class distribution:\n{pd.Series(y_resampled).value_counts().sort_index()}")
        
        return X_resampled, y_resampled
    
    def train_and_evaluate(self, X, y):
        """
        Train and evaluate four machine learning models using stratified k-fold cross-validation
        """
        print("\n=== MODEL TRAINING AND EVALUATION ===")
        
        # Initialize models
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'Gradient Boosting': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                                    random_state=self.random_state, eval_metric='mlogloss')
        }
        
        # Stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        results_summary = {
            'Algorithm': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC-ROC': [],
            'Std_Dev': []
        }
        
        self.models = models
        
        for model_name, model in models.items():
            print(f"\n--- {model_name} ---")
            
            fold_results = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            fold_num = 1
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Apply SMOTE only on training data
                smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                # Normalize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_resampled)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train_resampled)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                
                fold_results['accuracy'].append(acc)
                fold_results['precision'].append(prec)
                fold_results['recall'].append(rec)
                fold_results['f1'].append(f1)
                fold_results['auc'].append(auc_score)
                
                print(f"  Fold {fold_num}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc_score:.3f}")
                fold_num += 1
            
            # Calculate mean and std
            mean_acc = np.mean(fold_results['accuracy'])
            mean_prec = np.mean(fold_results['precision'])
            mean_rec = np.mean(fold_results['recall'])
            mean_f1 = np.mean(fold_results['f1'])
            mean_auc = np.mean(fold_results['auc'])
            std_dev = np.std(fold_results['accuracy'])
            
            results_summary['Algorithm'].append(model_name)
            results_summary['Accuracy'].append(mean_acc)
            results_summary['Precision'].append(mean_prec)
            results_summary['Recall'].append(mean_rec)
            results_summary['F1-Score'].append(mean_f1)
            results_summary['AUC-ROC'].append(mean_auc)
            results_summary['Std_Dev'].append(std_dev)
            
            print(f"\n  Average Performance:")
            print(f"    Accuracy: {mean_acc:.4f} (±{std_dev:.4f})")
            print(f"    Precision: {mean_prec:.4f}")
            print(f"    Recall: {mean_rec:.4f}")
            print(f"    F1-Score: {mean_f1:.4f}")
            print(f"    AUC-ROC: {mean_auc:.4f}")
        
        self.results = pd.DataFrame(results_summary)
        return self.results
    
    def plot_results(self, save_path='results/'):
        """Generate visualizations of model performance"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # 1. Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Machine Learning Models Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(self.results['Algorithm'], self.results[metric], color='steelblue', alpha=0.7)
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_performance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}model_performance_comparison.png")
        plt.close()
        
        # 2. Performance Table
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = self.results[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].copy()
        table_data['Accuracy'] = table_data['Accuracy'].apply(lambda x: f'{x:.4f}')
        table_data['Precision'] = table_data['Precision'].apply(lambda x: f'{x:.4f}')
        table_data['Recall'] = table_data['Recall'].apply(lambda x: f'{x:.4f}')
        table_data['F1-Score'] = table_data['F1-Score'].apply(lambda x: f'{x:.4f}')
        table_data['AUC-ROC'] = table_data['AUC-ROC'].apply(lambda x: f'{x:.4f}')
        
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                        cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Metrics Summary', fontweight='bold', fontsize=14, pad=20)
        plt.savefig(f'{save_path}performance_table.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}performance_table.png")
        plt.close()
    
    def display_results(self):
        """Display results in formatted table"""
        print("\n" + "="*80)
        print("FINAL RESULTS - MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(self.results.to_string(index=False))
        print("="*80)


def main():
    """Main execution function"""
    
    print("="*80)
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
    print("Machine Learning Project - COM7014")
    print("="*80)
    
    # Initialize predictor
    predictor = StudentPerformancePredictor(random_state=42)
    
    # For demonstration, we'll create sample data
    # In practice, replace this with actual UCI or OULAD dataset
    print("\n=== CREATING SAMPLE DATASET ===")
    print("Note: Using sample data for demonstration.")
    print("Replace with actual UCI or OULAD dataset for full analysis.")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    sample_data = pd.DataFrame({
        'Age': np.random.randint(18, 30, n_samples),
        'Prior_GPA': np.random.uniform(2.0, 4.0, n_samples),
        'Attendance_Rate': np.random.uniform(0.5, 1.0, n_samples),
        'Assignments_Submitted': np.random.randint(0, 30, n_samples),
        'Quiz_Score': np.random.uniform(0, 100, n_samples),
        'Study_Hours': np.random.uniform(0, 50, n_samples),
        'Class_Participation': np.random.randint(0, 100, n_samples),
        'Previous_Failures': np.random.randint(0, 3, n_samples),
        'Target_Grade': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    })
    
    # Preprocess
    X, y = predictor.preprocess_data(sample_data, 'Target_Grade')
    
    # Feature selection
    X_selected, selected_features, feature_importance = predictor.feature_selection(X, y)
    
    # Train and evaluate
    results = predictor.train_and_evaluate(X_selected, y)
    
    # Display results
    predictor.display_results()
    
    # Generate visualizations
    predictor.plot_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Output saved to 'results/' directory")
    print("\nNext Steps:")
    print("1. Replace sample data with actual UCI or OULAD dataset")
    print("2. Review feature importance rankings")
    print("3. Consider Random Forest or XGBoost for deployment")
    print("4. Implement early intervention strategies based on predictions")


if __name__ == "__main__":
    main()
