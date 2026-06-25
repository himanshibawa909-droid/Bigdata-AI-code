"""
Data Loading Guide - UCI and OULAD Datasets

This file shows how to load and prepare real datasets for the 
student performance prediction project.

Options:
1. Download from UCI Machine Learning Repository
2. Download from Open University
3. Use your own institutional data
4. Use the sample data provided
"""

import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
import zipfile

# ============================================================================
# OPTION 1: Load UCI Student Performance Dataset
# ============================================================================

def load_uci_dataset():
    """
    Download and load UCI Student Performance Dataset
    
    Dataset info:
    - Source: UCI Machine Learning Repository
    - URL: https://archive.ics.uci.edu/dataset/320/student+performance
    - Records: ~395 students
    - Features: 33 (student grades, demographics, school info)
    - Target: Final grade (G3)
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    
    print("Loading UCI Student Performance Dataset...")
    print("-" * 60)
    
    # Option A: Download from URL (if not already downloaded)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    filename = "uci_student_data.zip"
    
    try:
        if not os.path.exists("student_data"):
            print("Downloading dataset (first time only)...")
            urlretrieve(url, filename)
            
            # Extract
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall("student_data")
            print("✓ Download complete")
        
        # Load the data
        # UCI has two CSV files: student-mat.csv (Math) and student-por.csv (Portuguese)
        # We'll use the Math dataset
        df = pd.read_csv("student_data/student-mat.csv", sep=";")
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {df.shape[0]} students, {df.shape[1]} features")
        print(f"\nFeatures:")
        print(f"  - Demographics: age, sex, address, famsize, Pstatus")
        print(f"  - Parent info: Medu, Fedu, Mjob, Fjob")
        print(f"  - Education: schoolsup, famsup, paid, activities")
        print(f"  - Lifestyle: alcohol, freetime, goout, Dalc, Walc")
        print(f"  - Academics: failures, schoolsup, famsup, absences")
        print(f"  - Grades: G1 (1st period), G2 (2nd period), G3 (final grade)")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading UCI dataset: {e}")
        print(f"  Try downloading manually from:")
        print(f"  https://archive.ics.uci.edu/dataset/320/student+performance")
        return None


# ============================================================================
# OPTION 2: Load Open University Learning Analytics Dataset (OULAD)
# ============================================================================

def load_oulad_dataset():
    """
    Load OULAD (Open University Learning Analytics Dataset)
    
    Dataset info:
    - Source: Open University, UK
    - URL: https://analyse.kmi.open.ac.uk/open_dataset
    - Records: ~32,000 student registrations
    - Features: Multiple files with student, course, interaction data
    - Note: Must register to download
    
    Returns:
        pandas.DataFrame: Merged dataset with student data and interactions
    """
    
    print("Loading OULAD Dataset...")
    print("-" * 60)
    
    print("Manual Download Required:")
    print("1. Visit: https://analyse.kmi.open.ac.uk/open_dataset")
    print("2. Register for access")
    print("3. Download OULAD files")
    print("4. Extract to 'oulad_data/' directory")
    print("\nExpected files:")
    print("  - studentInfo.csv (student demographics & performance)")
    print("  - studentRegistration.csv (enrollment info)")
    print("  - studentVle.csv (Virtual Learning Environment interactions)")
    print("  - courses.csv (course information)")
    
    try:
        # Load main student info
        student_info = pd.read_csv("oulad_data/studentInfo.csv")
        
        # Load VLE interactions (engagement)
        student_vle = pd.read_csv("oulad_data/studentVle.csv")
        
        # Aggregate VLE data by student
        vle_summary = student_vle.groupby('id_student').agg({
            'date': 'count',  # number of interactions
            'sum_click': 'sum'  # total clicks
        }).rename(columns={'date': 'num_interactions', 'sum_click': 'total_clicks'})
        
        # Merge
        df = student_info.merge(vle_summary, left_on='id_student', right_index=True, how='left')
        
        # Fill missing values
        df['num_interactions'] = df['num_interactions'].fillna(0)
        df['total_clicks'] = df['total_clicks'].fillna(0)
        
        print(f"✓ OULAD dataset loaded successfully")
        print(f"  Shape: {df.shape[0]} student registrations, {df.shape[1]} features")
        print(f"\nFeatures available:")
        print(f"  - Demographics: gender, age_band, region, highest_education")
        print(f"  - Academic: code_module, code_presentation, num_of_prev_attempts")
        print(f"  - Outcome: final_result (Pass/Fail/Distinction)")
        print(f"  - Engagement: num_interactions, total_clicks")
        
        return df
        
    except FileNotFoundError as e:
        print(f"✗ OULAD files not found: {e}")
        print(f"  Please download from: https://analyse.kmi.open.ac.uk/open_dataset")
        return None


# ============================================================================
# OPTION 3: Prepare UCI Dataset for Modeling
# ============================================================================

def prepare_uci_for_modeling(df):
    """
    Prepare UCI Student Performance Data for machine learning
    
    Args:
        df: Raw UCI dataset
        
    Returns:
        pandas.DataFrame: Cleaned and processed dataset
    """
    
    print("\nPreparing UCI data for modeling...")
    
    # Create binary target variable
    # Passing grade: G3 >= 10 (on scale of 0-20)
    df_processed = df.copy()
    
    # Option 1: Binary classification (Pass/Fail)
    df_processed['performance'] = (df_processed['G3'] >= 10).astype(int)
    df_processed.rename(columns={'performance': 'Target_Performance'}, inplace=True)
    
    # Option 2: Multi-class classification
    # df_processed['performance'] = pd.cut(df_processed['G3'], 
    #                                      bins=[-1, 9, 14, 20],
    #                                      labels=['Low', 'Medium', 'High'])
    
    # Drop original grades (we only use final grade as target)
    df_processed = df_processed.drop(['G1', 'G2', 'G3'], axis=1)
    
    # Select relevant features
    relevant_features = [
        'age', 'absences', 'failures', 'studytime', 'freetime',
        'goout', 'Dalc', 'Walc', 'health', 'Medu', 'Fedu',
        'Target_Performance'
    ]
    
    df_processed = df_processed[relevant_features]
    
    print(f"✓ Data prepared")
    print(f"  Features: {len(df_processed.columns) - 1}")
    print(f"  Samples: {len(df_processed)}")
    print(f"  Target distribution: {df_processed['Target_Performance'].value_counts().to_dict()}")
    
    return df_processed


# ============================================================================
# OPTION 4: Prepare OULAD Dataset for Modeling
# ============================================================================

def prepare_oulad_for_modeling(df):
    """
    Prepare OULAD data for machine learning
    
    Args:
        df: Raw OULAD dataset
        
    Returns:
        pandas.DataFrame: Cleaned and processed dataset
    """
    
    print("\nPreparing OULAD data for modeling...")
    
    df_processed = df.copy()
    
    # Convert categorical outcome to binary
    # Distinction/Pass = 1 (Success), Withdrawn/Fail = 0 (Not successful)
    outcome_mapping = {
        'Withdrawn': 0,
        'Fail': 0,
        'Pass': 1,
        'Distinction': 1
    }
    
    df_processed['Target_Performance'] = df_processed['final_result'].map(outcome_mapping)
    
    # One-hot encode categorical features
    categorical_cols = ['gender', 'age_band', 'highest_education', 'region']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Select relevant features
    feature_cols = [col for col in df_processed.columns 
                   if col not in ['id_student', 'final_result', 'Target_Performance']]
    
    df_processed = df_processed[feature_cols + ['Target_Performance']]
    
    # Remove rows with missing target
    df_processed = df_processed.dropna(subset=['Target_Performance'])
    
    print(f"✓ Data prepared")
    print(f"  Features: {len(df_processed.columns) - 1}")
    print(f"  Samples: {len(df_processed)}")
    print(f"  Target distribution: {df_processed['Target_Performance'].value_counts().to_dict()}")
    
    return df_processed


# ============================================================================
# OPTION 5: Load Local CSV File
# ============================================================================

def load_local_dataset(filepath):
    """
    Load dataset from local CSV file
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded from {filepath}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return None


# ============================================================================
# OPTION 6: Generate Sample Dataset
# ============================================================================

def generate_sample_dataset(n_students=500, random_state=42):
    """
    Generate synthetic student dataset for testing
    
    Args:
        n_students (int): Number of students
        random_state (int): Random seed
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    
    np.random.seed(random_state)
    
    data = {
        'Student_ID': range(1, n_students + 1),
        'Age': np.random.randint(18, 30, n_students),
        'Prior_GPA': np.random.uniform(1.5, 4.0, n_students),
        'Attendance_Rate': np.random.uniform(0.40, 1.0, n_students),
        'Study_Hours_Per_Week': np.random.uniform(0, 50, n_students),
        'Assignment_Submission_Rate': np.random.uniform(0.30, 1.0, n_students),
        'Quiz_Average': np.random.uniform(30, 100, n_students),
        'LMS_Logins': np.random.randint(5, 200, n_students),
        'Previous_Failures': np.random.randint(0, 4, n_students),
        'Parent_Education_Level': np.random.randint(1, 5, n_students),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features
    # Students with good GPA, attendance, and engagement tend to pass
    probability_pass = (
        (df['Prior_GPA'] / 4.0) * 0.3 +
        df['Attendance_Rate'] * 0.3 +
        (df['Quiz_Average'] / 100) * 0.2 +
        (1 - df['Previous_Failures'] / 4) * 0.2
    )
    
    df['Target_Grade'] = (np.random.random(n_students) < probability_pass).astype(int)
    df['Target_Grade'] = df['Target_Grade'].map({0: 'Fail', 1: 'Pass'})
    
    print(f"✓ Sample dataset generated")
    print(f"  Students: {n_students}")
    print(f"  Features: {len(df.columns) - 2}")
    print(f"  Target distribution: {df['Target_Grade'].value_counts().to_dict()}")
    
    return df


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("DATA LOADING GUIDE - Student Performance Prediction")
    print("="*60)
    
    # Example 1: Generate sample data (fastest, no download needed)
    print("\n[Example 1] Using Sample Dataset")
    print("-" * 60)
    df_sample = generate_sample_dataset(n_students=500)
    print(f"\nFirst few rows:")
    print(df_sample.head())
    
    # Example 2: Load UCI dataset
    # print("\n[Example 2] Using UCI Dataset")
    # print("-" * 60)
    # df_uci = load_uci_dataset()
    # if df_uci is not None:
    #     df_uci_prepared = prepare_uci_for_modeling(df_uci)
    #     print(f"\nFirst few rows:")
    #     print(df_uci_prepared.head())
    
    # Example 3: Load OULAD dataset
    # print("\n[Example 3] Using OULAD Dataset")
    # print("-" * 60)
    # df_oulad = load_oulad_dataset()
    # if df_oulad is not None:
    #     df_oulad_prepared = prepare_oulad_for_modeling(df_oulad)
    #     print(f"\nFirst few rows:")
    #     print(df_oulad_prepared.head())
    
    # Example 4: Load from local file
    # print("\n[Example 4] Loading from Local File")
    # print("-" * 60)
    # df_local = load_local_dataset('your_data.csv')
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Choose a dataset (sample, UCI, OULAD, or your own)")
    print("2. Load using appropriate function above")
    print("3. Pass to StudentPerformancePredictor for modeling")
    print("="*60)


# ============================================================================
# QUICK START CODE
# ============================================================================

"""
# Quick Start Example:

from student_performance_prediction import StudentPerformancePredictor
from data_loading import generate_sample_dataset

# 1. Load data
df = generate_sample_dataset(n_students=500)

# 2. Initialize predictor
predictor = StudentPerformancePredictor(random_state=42)

# 3. Preprocess
X, y = predictor.preprocess_data(df, target_column='Target_Grade')

# 4. Feature selection
X_selected, features, importance = predictor.feature_selection(X, y, top_n=8)

# 5. Train and evaluate
results = predictor.train_and_evaluate(X_selected, y)

# 6. Display results
predictor.display_results()
predictor.plot_results()
"""
