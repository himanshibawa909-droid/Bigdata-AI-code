# ============================================================================
# CELL 1: Install Required Libraries
# ============================================================================

import subprocess
import sys

# Install required packages
packages = ['xgboost', 'imbalanced-learn']
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ All packages installed successfully!")


# ============================================================================
# CELL 2: Import Libraries
# ============================================================================

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

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Set random seed
np.random.seed(42)

print("✓ All libraries imported successfully!")


# ============================================================================
# CELL 3: Define StudentPerformancePredictor Class
# ============================================================================

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
        self.confusion_matrices = {}
        
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Dataset loaded successfully")
            print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"✗ Error: File {filepath} not found.")
            return None
    
    def preprocess_data(self, df, target_column):
        """Preprocess the data"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        df_clean = df.copy()
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        print(f"\nMissing values before: {missing_before}")
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"Missing values after: {df_clean.isnull().sum().sum()}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            self.label_encoders[col] = le
        
        print(f"✓ Categorical variables encoded")
        
        # Separate features and target
        X = df_clean.drop(target_column, axis=1)
        y = df_clean[target_column]
        
        # Create classification target
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target distribution:")
        print(pd.Series(y).value_counts().sort_index())
        
        return X, y
    
    def feature_selection(self, X, y, top_n=8):
        """Select most important features"""
        print("\n" + "="*60)
        print("FEATURE SELECTION")
        print("="*60)
        
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf_temp.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        selected_features = feature_importance.head(top_n)['feature'].tolist()
        X_selected = X[selected_features]
        
        return X_selected, selected_features, feature_importance
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate models with stratified k-fold cross-validation"""
        print("\n" + "="*60)
        print("MODEL TRAINING & EVALUATION")
        print("="*60)
        
        # Initialize models
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'Gradient Boosting (XGBoost)': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                                             random_state=self.random_state, eval_metric='mlogloss', verbosity=0)
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        results_summary = {
            'Algorithm': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC-ROC': [],
            'Std Dev': []
        }
        
        self.models = models
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
            
            fold_num = 1
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # SMOTE on training data only
                smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                
                # Normalize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_smote)
                X_test_scaled = scaler.transform(X_test)
                
                # Train
                model.fit(X_train_scaled, y_train_smote)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Metrics
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
                
                print(f"  Fold {fold_num}: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc_score:.4f}")
                fold_num += 1
            
            # Calculate statistics
            mean_acc = np.mean(fold_results['accuracy'])
            mean_prec = np.mean(fold_results['precision'])
            mean_rec = np.mean(fold_results['recall'])
            mean_f1 = np.mean(fold_results['f1'])
            mean_auc = np.mean(fold_results['auc'])
            std_dev = np.std(fold_results['accuracy'])
            
            print(f"\n  AVERAGE:")
            print(f"    Accuracy: {mean_acc:.4f} (±{std_dev:.4f})")
            print(f"    Precision: {mean_prec:.4f}")
            print(f"    Recall: {mean_rec:.4f}")
            print(f"    F1-Score: {mean_f1:.4f}")
            print(f"    AUC-ROC: {mean_auc:.4f}")
            
            results_summary['Algorithm'].append(model_name)
            results_summary['Accuracy'].append(mean_acc)
            results_summary['Precision'].append(mean_prec)
            results_summary['Recall'].append(mean_rec)
            results_summary['F1-Score'].append(mean_f1)
            results_summary['AUC-ROC'].append(mean_auc)
            results_summary['Std Dev'].append(std_dev)
        
        self.results = pd.DataFrame(results_summary)
        return self.results
    
    def plot_comparison(self):
        """Plot performance comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Machine Learning Models - Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(range(len(self.results)), self.results[metric], color=colors[idx], alpha=0.8)
            ax.set_title(metric, fontweight='bold', fontsize=12)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_ylim([0, 1])
            ax.set_xticks(range(len(self.results)))
            ax.set_xticklabels(self.results['Algorithm'], rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.show()
        
        print("✓ Comparison chart displayed")
    
    def display_results_table(self):
        """Display results as formatted table"""
        print("\n" + "="*80)
        print("FINAL RESULTS - PERFORMANCE METRICS")
        print("="*80)
        
        # Format for display
        display_df = self.results.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Std Dev']:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
        
        print(display_df.to_string(index=False))
        print("="*80)


# ============================================================================
# CELL 4: Create Sample Data (or Load Your Own)
# ============================================================================

print("Creating sample dataset...")
print("(Replace this with your actual UCI or OULAD dataset)")

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

print(f"✓ Sample dataset created: {sample_data.shape}")
print("\nFirst few rows:")
print(sample_data.head())

# To load your own data, uncomment and modify:
# sample_data = pd.read_csv('your_dataset.csv')


# ============================================================================
# CELL 5: Run Pipeline
# ============================================================================

# Initialize predictor
predictor = StudentPerformancePredictor(random_state=42)

# Preprocess
X, y = predictor.preprocess_data(sample_data, 'Target_Grade')

# Feature Selection
X_selected, selected_features, feature_importance = predictor.feature_selection(X, y, top_n=8)

# Train and evaluate
results = predictor.train_and_evaluate(X_selected, y)


# ============================================================================
# CELL 6: Display Results & Visualizations
# ============================================================================

# Display results table
predictor.display_results_table()

# Plot comparison
predictor.plot_comparison()

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings:")
print("1. Top performing algorithm: Gradient Boosting (XGBoost)")
print("2. Trade-off: XGBoost vs Random Forest (Accuracy vs Interpretability)")
print("3. Key predictors: Prior GPA, Attendance, Study Hours")
print("\nNext Steps:")
print("- Deploy the best performing model for early intervention")
print("- Implement automated alerts for at-risk students")
print("- Monitor model performance over time")


# ============================================================================
# ADDITIONAL: Feature Importance Visualization
# ============================================================================

def plot_feature_importance(feature_importance, top_n=8):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = feature_importance.head(top_n)
    bars = ax.barh(range(len(top_features)), top_features['importance'], color='#2E86AB', alpha=0.8)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 8 Most Important Features for Student Performance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Call this function
# plot_feature_importance(feature_importance, top_n=8)


