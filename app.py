import os;
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import requests # Import the requests library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef

# 1. Set the Streamlit page configuration
st.set_page_config(
    page_title='Direct Marketing Campaigns on Portuguese Banking Data',
    page_icon='💰',
    layout='wide'
)


# Set 'Light Gray' as the default selected color
color_options = {
    'Light Gray': '#D3D3D3',
    'Black': '#000000'
}

# 2. Set the main title of the Streamlit application
st.title('💰 Direct Marketing Campaigns on Portuguese Banking Data')
st.markdown("""
This Application evaluates various machine learning models to accurately predict
whether a client will subscribe to a term deposit, based on the data from a direct
marketing campaign conducted by a Portuguese banking institution.
""")

# Add developer header
st.header('Developed by Vaibhav Khare - BITS ID: 2025ab05182@wilp.bits-pilani.ac.in')

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")



# Initialize session state variables if they don't exist
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'y_pred' not in st.session_state:
    st.session_state['y_pred'] = None
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = 'Logistic Regression' # Default selection
if 'original_df' not in st.session_state: # Initialize original_df
    st.session_state['original_df'] = None

# Callback function to reset model and predictions when model selection changes
def reset_model_and_results():
    st.session_state['model'] = None
    st.session_state['y_pred'] = None
    st.info("Model and prediction results cleared due to model selection change.")

# --- Data Upload ---
st.header('📤 Data Upload - Upload Your Test Data')
uploaded_file = st.file_uploader("👆 Upload your CSV file", type=["csv"])

# --- Download Sample Data ---
st.subheader("📥 Download sample test dataset if dataset is not available:")
sample_data_url = 'https://raw.githubusercontent.com/vaibhavkhare1206/ML-Assignment-Repo-2025ab05182/main/data/bank-test.csv'

try:
    response = requests.get(sample_data_url)
    if response.status_code == 200:
        st.download_button(
            label="Download Sample 'bank-test.csv'",
            data=response.content,
            file_name="bank-test.csv",
            mime="text/csv"
        )
    else:
        st.error(f"❌ Failed to fetch sample data from URL. Status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    st.error(f"Error fetching sample data: {e}")



if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully! Here's a preview of your data:")
        st.dataframe(df.head())
        st.session_state['original_df'] = df.copy() # Store original for potential re-preprocessing
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("👆Please upload a CSV file to get started.")

# --- Data Preprocessing ---
st.header('🗄️Data Preprocessing')
if st.session_state['original_df'] is not None:
    df = st.session_state['original_df'].copy()

    # Convert target variable 'y' from 'yes'/'no' to 1/0
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        st.write("🎯 Target variable 'y' converted to numerical (1/0).")
    else:
        st.warning("🎯 Target variable 'y' not found. Ensure the dataset contains a 'y' column.")

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Remove 'y' from categorical_cols if it's there (after conversion it's numerical)
    if 'y' in categorical_cols:
        categorical_cols.remove('y')

    # Apply one-hot encoding to categorical columns
    if categorical_cols:
        df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        st.write(f"One-hot encoding applied to categorical columns: {', '.join(categorical_cols)}.")
    else:
        df_processed = df.copy()
        st.info("❌ No categorical columns found for one-hot encoding.")

    # --- Add Missing Value Handling ---
    st.subheader("❓Handling Missing Values")

    # Identify columns with NaN values
    missing_cols_nan = df_processed.columns[df_processed.isnull().any()].tolist()

    # Identify columns with infinite values (only check numerical columns)
    numerical_cols_in_processed_df = df_processed.select_dtypes(include=np.number).columns
    missing_cols_inf = [col for col in numerical_cols_in_processed_df if np.isinf(df_processed[col]).any()]

    if missing_cols_nan or missing_cols_inf:
        st.warning("⚠️ Found missing or infinite values. Handling them...")

        # Handle NaN values
        if missing_cols_nan:
            st.write(f"Imputing NaN values in columns: {', '.join(missing_cols_nan)} with mean/mode.")
            for col in missing_cols_nan:
                if df_processed[col].dtype in ['int64', 'float64']:
                    # Calculate mean, if mean is NaN (meaning all values are NaN), use 0 as fallback
                    imputation_value = df_processed[col].mean()
                    if pd.isna(imputation_value):
                        imputation_value = 0 # Fallback for entirely NaN numerical columns
                    df_processed[col].fillna(imputation_value, inplace=True)
                else:
                    # For non-numeric NaNs, use mode. Handle case where mode might be empty (e.g., all NaN)
                    if not df_processed[col].mode().empty:
                        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    else:
                        df_processed[col].fillna('unknown', inplace=True) # Fallback for entirely NaN object columns

        # Handle infinite values
        if missing_cols_inf:
            st.write(f"Replacing infinite values in columns: {', '.join(missing_cols_inf)} with NaN, then re-imputing with mean.")
            for col in missing_cols_inf:
                # Replace inf with NaN first
                df_processed[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                # Re-impute if it was originally inf, now NaN. Use 0 as fallback if mean is NaN.
                if df_processed[col].isnull().any():
                     imputation_value = df_processed[col].mean()
                     if pd.isna(imputation_value):
                         imputation_value = 0 # Fallback for entirely NaN numerical columns after inf replacement
                     df_processed[col].fillna(imputation_value, inplace=True)

        st.success("🟩 Missing and infinite values handled.")
    else:
        st.info("✅ No missing or infinite values found.")

    st.subheader("💽 Preprocessed Data Preview:")
    st.dataframe(df_processed.head())
    st.session_state['df_processed'] = df_processed

# --- Data Splitting ---
st.header('🪓Data Splitting')
if st.session_state['df_processed'] is not None:
    df_processed = st.session_state['df_processed']

    if 'y' in df_processed.columns:
        X = df_processed.drop('y', axis=1)
        y = df_processed['y']
        st.write("Features (X) and target (y) defined.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("🪓Data split into training and testing sets.")

        st.subheader("🔸Shapes of Training and Testing Sets:")
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"y_train shape: {y_train.shape}")
        st.write(f"X_test shape: {X_test.shape}")
        st.write(f"y_test shape: {y_test.shape}")

        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

    else:
        st.error("Target variable 'y' not found in the processed DataFrame. Cannot split data.")

# --- Model Selection ---
st.sidebar.header('📊 Model Selection')
model_options = [
    'Logistic Regression',
    'Decision Tree Classifier',
    'K-Nearest Neighbor Classifier',
    'Naive Bayes Classifier (Gaussian Model)',
    'Random Forest Model',
    'XGBoost Model'
]

st.session_state['selected_model'] = st.sidebar.selectbox(
    'Choose a Classification Model',
    model_options,
    index=model_options.index(st.session_state['selected_model']),
    key='model_selector', # Added a unique key
    on_change=reset_model_and_results # Added the callback function
)
st.sidebar.write(f"You selected the **{st.session_state['selected_model']}** model.")

# --- Model Training ---
st.header('🍥 Model Training')
if st.session_state['X_train'] is not None and st.session_state['y_train'] is not None:
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']

    if st.button('🍥Train Model'):
        st.info(f"Training {st.session_state['selected_model']}...")
        try:
            model = None
            if st.session_state['selected_model'] == 'Logistic Regression':
                model = LogisticRegression(random_state=42, solver='liblinear')
            elif st.session_state['selected_model'] == 'Decision Tree Classifier':
                model = DecisionTreeClassifier(random_state=42)
            elif st.session_state['selected_model'] == 'K-Nearest Neighbor Classifier':
                model = KNeighborsClassifier()
            elif st.session_state['selected_model'] == 'Naive Bayes Classifier (Gaussian Model)':
                model = GaussianNB()
            elif st.session_state['selected_model'] == 'Random Forest Model':
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
            elif st.session_state['selected_model'] == 'XGBoost Model':
                model = xgb.XGBClassifier(n_estimators=5, random_state=42, use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')

            if model:
                model.fit(X_train, y_train);
                st.success(f"{st.session_state['selected_model']} trained successfully!") 
                st.session_state['model'] = model
                st.session_state['y_pred'] = model.predict(X_test)

                # Modified block to handle AttributeError robustly
                if st.session_state['selected_model'] == 'Random Forest Model':
                    try:
                        num_estimators = len(model.estimators_)
                        st.info(f"Number of estimators (trees) in the Random Forest model: {num_estimators}")
                    except AttributeError:
                        st.warning("❌ Could not access 'estimators_' attribute for Random Forest model after training. Ensure the model fitted successfully.")
            else:
                st.error("❌ No model selected or initialized.")
        except Exception as e:
            st.error(f"Error training model: {e}")

    if st.session_state['model'] is not None:
        st.success("✅ Model ready for evaluation.")
    else:
        st.info("Click 'Train Model' to begin.")
else:
    st.warning("Please upload data and complete preprocessing/splitting steps to train a model.")

# --- Model Evaluation ---
st.header('🎯 Model Evaluation - Prediction Report Generated Successfully ✅ ')
if st.session_state['model'] is not None and st.session_state['y_test'] is not None and st.session_state['y_pred'] is not None:
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']

    cm_df = None # Initialize cm_df for robustness

    if y_test.empty or len(y_pred) == 0:
        st.warning("y_test or y_pred is empty. Cannot evaluate an empty set.")
    else:
        st.write(f"Evaluating {st.session_state['selected_model']}:")

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # Ensure predict_proba is available and model is fitted for AUC score
        if hasattr(st.session_state['model'], 'predict_proba'):
            try:
                auc_score = roc_auc_score(y_test, st.session_state['model'].predict_proba(st.session_state['X_test'])[:, 1])
            except Exception as e:
                st.warning(f"❌ Could not calculate AUC score: {e}. Some models may not support predict_proba or require specific data types.")
                auc_score = 'N/A'
        else:
            st.warning("❌ Selected model does not have a 'predict_proba' method for AUC calculation.")
            auc_score = 'N/A'

        mcc = matthews_corrcoef(y_test, y_pred)

        # Display metrics in a table
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score', 'Matthews Correlation Coefficient (MCC)'] ,
            'Value': [accuracy, precision, recall, f1, auc_score, mcc]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.subheader("📇Evaluation Matrix")
        st.dataframe(metrics_df.set_index('Metric'))

        # Display Classification Report
        st.subheader("📇Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Display Confusion Matrix (as DataFrame)
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=['Actual Negative (0)', 'Actual Positive (1)'], columns=['Predicted Negative (0)', 'Predicted Positive (1)'])
        st.dataframe(cm_df)

        # Plot the confusion matrix using seaborn, only if cm_df was created
        if cm_df is not None and not cm_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cm_df.columns,
                    yticklabels=cm_df.index,
                    ax=ax, cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# --- Reset Button ---
st.sidebar.markdown("---")
if st.sidebar.button('Clear Results and Reset'):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
