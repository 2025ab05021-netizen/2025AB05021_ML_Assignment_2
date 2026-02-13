import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.datasets import load_breast_cancer


st.set_page_config(page_title="Machine Learning classification models", layout="centered")


def load_models():
    models = {}
    names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'xgboost']
    for n in names:
        try:
            models[n] = joblib.load(f"{n}_model.pkl")
        except Exception:
            models[n] = None
    try:
        scaler = joblib.load('scaler.pkl')
    except Exception:
        scaler = None
    try:
        with open('metadata.pkl','rb') as f:
            metadata = pickle.load(f)
    except Exception:
        metadata = None
    return models, scaler, metadata


def load_model_results():
    try:
        return pd.read_csv('model_results.csv')
    except Exception:
        return None


def auto_detect_target_column(df):
    if df is None:
        return None
    for c in df.columns:
        if c.lower() in ['target','label','class','y','outcome']:
            return c
    for c in df.columns:
        if df[c].nunique() == 2:
            return c
    return df.columns[-1]


def prepare_data_for_prediction(df, target_col, scaler):
    if df is None:
        return None, None
    dfc = df.copy()
    if target_col in dfc.columns:
        y = dfc[target_col]
        X = dfc.drop(columns=[target_col])
    else:
        y = None
        X = dfc
    
    #  Data preprocessing
    for col in X.columns:
        if X[col].dtype == object:
            # Try to convert to numeric first
            X[col] = pd.to_numeric(X[col], errors='coerce')
            # If still object, do label encoding for categorical
            if X[col].dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    
    # Convert to numpy array
    X_values = X.values
    
    # Apply scaling if available
    try:
        if scaler is not None:
            X_values = scaler.transform(X_values)
    except Exception as e:
        try:
            pass
        except Exception:
            pass
    
    return X_values, y

def validate_dataset_requirements(df, target_col):
    """
    Validate dataset meets minimum requirements but allow processing anyway
    """
    if df is None:
        return False, "No dataset provided", []
    
    issues = []
    warnings = []
    
    # Get feature count
    if target_col in df.columns:
        n_features = len(df.columns) - 1
        X = df.drop(columns=[target_col])
    else:
        n_features = len(df.columns)
        X = df
    
    n_instances = len(df)
    
    # Check minimum requirements
    min_features = 12
    min_instances = 500
    
    # Feature count check
    if n_features < min_features:
        warnings.append(f"⚠️ Dataset has {n_features} features (recommended: ≥{min_features}). Predictions may be less accurate.")
    else:
        issues.append(f"✅ Features: {n_features}/{min_features} (Good)")
    
    # Instance count check
    if n_instances < min_instances:
        warnings.append(f"⚠️ Dataset has {n_instances} instances (recommended: ≥{min_instances}). Model performance may vary.")
    else:
        issues.append(f"✅ Instances: {n_instances}/{min_instances} (Good)")
    
    # Additional data quality checks
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        warnings.append(f"ℹ️ Dataset has {missing_values} missing values (will be auto-filled).")
    else:
        issues.append(f"✅ No missing values detected")
    
    # Check for categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        warnings.append(f"ℹ️ {len(categorical_cols)} categorical columns detected (will be auto-encoded).")
    
    return True, "Dataset validation completed", issues + warnings

def get_model_display_name_mapping():
    """
    Map internal model names to display names and CSV names
    """
    return {
        'logistic_regression': {
            'display': 'Logistic Regression',
            'csv_pattern': 'Logistic Regression'
        },
        'decision_tree': {
            'display': 'Decision Tree',
            'csv_pattern': 'Decision Tree'
        },
        'knn': {
            'display': 'K-Nearest Neighbors',
            'csv_pattern': 'K-Nearest Neighbors'
        },
        'naive_bayes': {
            'display': 'Naive Bayes',
            'csv_pattern': 'Naive Bayes'
        },
        'random_forest': {
            'display': 'Random Forest',
            'csv_pattern': 'Random Forest'
        },
        'xgboost': {
            'display': 'XGBoost',
            'csv_pattern': 'XGBoost'
        }
    }

def align_features(X, model, metadata=None, strategy='pad_truncate'):
    """
    Align input features with model expectations using various strategies
    
    Strategies:
    - 'pad_truncate': Add zeros for missing features, truncate extra features
    - 'select_common': Use only features that match (requires feature names)
    - 'warning_only': Show warning but allow prediction
    """
    if X is None:
        return None, "No data provided", False
    
    n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0]) if X else 0
    
    expected_features = None
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
    elif metadata and 'n_features' in metadata:
        expected_features = metadata['n_features']
    
    if expected_features is None:
        return X, f"Model feature requirements unknown. Using {n_features} features as-is.", True
    
    if n_features == expected_features:
        return X, f"Perfect match: {n_features} features", True
    
    # Handle feature mismatch based on strategy
    if strategy == 'pad_truncate':
        if n_features < expected_features:
            padding = np.zeros((X.shape[0], expected_features - n_features))
            X_aligned = np.hstack([X, padding])
            message = f"Added {expected_features - n_features} zero-padding features ({n_features} → {expected_features})"
        else:
            X_aligned = X[:, :expected_features]
            message = f"Truncated {n_features - expected_features} extra features ({n_features} → {expected_features})"
        return X_aligned, message, True
        
    elif strategy == 'warning_only':
        message = f"Feature count mismatch: {n_features} vs {expected_features} expected. Predictions may be unreliable."
        return X, message, True
    
    else:
        message = f"Feature mismatch: {n_features} features provided, {expected_features} expected"
        return X, message, False

def load_default_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def main():
    st.title("ML classification models")
    st.markdown("""This web app evaluates multiple trained classification models; upload test data, choose a model, and view metrics, reports, and confusion matrices.""")
    st.markdown("- Upload a CSV test dataset\n- Select a trained model from a dropdown\n- View evaluation metrics (Accuracy, Precision, Recall, F1, AUC, MCC if available)\n- See Confusion Matrix and Classification Report")

    models, scaler, metadata = load_models()
    results_df = load_model_results()

    # 1) Dataset upload
    uploaded = st.file_uploader("Upload CSV test dataset.", type=['csv'])
    if uploaded is not None:
        try:
            # try default CSV parsing first
            try:
                test_df = pd.read_csv(uploaded)
            except Exception:
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
                test_df = pd.read_csv(uploaded, sep=None, engine='python')

            if test_df.shape[1] == 1:
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
                try:
                    test_df = pd.read_csv(uploaded, sep='\t')
                except Exception:
                    try:
                        uploaded.seek(0)
                    except Exception:
                        pass
                    test_df = pd.read_csv(uploaded, delim_whitespace=True)

            st.success(f"Loaded {len(test_df)} rows")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
            
        # detect target column
        detected = auto_detect_target_column(test_df)
        # normalize column names (strip whitespace)
        test_df.columns = [c.strip() for c in test_df.columns]
        if detected not in test_df.columns:
            lower_map = {c.lower(): c for c in test_df.columns}
            if detected and detected.lower() in lower_map:
                detected = lower_map[detected.lower()]
            else:
                detected = test_df.columns[-1]

       
        st.markdown(f"**Detected target column:** {detected}")
       
        target_col = st.selectbox("Select target column", options=list(test_df.columns), index=list(test_df.columns).index(detected) if detected in test_df.columns else 0)

       
        try:
            if target_col in test_df.columns:
                _u = test_df[target_col].dropna().unique()
                if len(_u) > 2:
                    test_df[target_col] = (pd.to_numeric(test_df[target_col], errors='coerce') > 0).astype(int)
        except Exception:
            pass
        
    else:
        test_df = load_default_dataset()
        target_col = 'target'

    # 2) Model selection
    available = [k for k,v in models.items() if v is not None]
    if not available:
        st.error('No trained models found. Place <name>_model.pkl files in the app folder.')
        return
    
    name_mapping = get_model_display_name_mapping()
    display = [name_mapping.get(k, {}).get('display', k.replace('_',' ').title()) for k in available]
    sel = st.selectbox('Select model', options=display)
    sel_key = available[display.index(sel)]
    model = models[sel_key]

    # 3) Display evaluation metrics 
    st.header('Evaluation Metrics')
    
    # Calculate live metrics for current dataset
    try:
        X_test, y_test = prepare_data_for_prediction(test_df, target_col, scaler)
        if X_test is not None:
            # Align features with model expectations
            X_aligned, alignment_msg, can_predict = align_features(X_test, model, metadata, 'pad_truncate')
            
            if can_predict:
                # Make prediction
                y_pred = model.predict(X_aligned)
                
                # Prepare labels for binary classification
                y_true = pd.Series(y_test).reset_index(drop=True)
                y_pred_s = pd.Series(y_pred).reset_index(drop=True)
                
                # Convert to binary if needed
                if y_true.nunique() > 2:
                    top = sorted(y_true.unique())[-1]
                    y_true_bin = (y_true == top).astype(int)
                    y_pred_bin = (y_pred_s == top).astype(int)
                else:
                    y_true_bin = y_true.astype(int)
                    y_pred_bin = y_pred_s.astype(int)
                
                # Calculate all metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                live_accuracy = accuracy_score(y_true_bin, y_pred_bin)
                live_precision = precision_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
                live_recall = recall_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
                live_f1 = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
                
                # Calculate AUC
                live_auc = 'N/A'
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_aligned)[:,1]
                    elif hasattr(model, 'decision_function'):
                        probs = model.decision_function(X_aligned)
                    else:
                        probs = None
                    if probs is not None and len(set(y_true_bin))==2:
                        live_auc = roc_auc_score(y_true_bin, probs)
                except Exception:
                    live_auc = 'N/A'
                    
                # Calculate MCC
                live_mcc = 'N/A'
                try:
                    live_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
                except Exception:
                    live_mcc = 'N/A'
                
                # Display metrics
                dataset_type = "" if uploaded is None else " - Uploaded Dataset"
                st.success(f"📊 Live Evaluation Metrics {dataset_type}")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric('Accuracy', f"{live_accuracy:.4f}")
                    st.metric('Precision', f"{live_precision:.4f}")
                with c2:
                    st.metric('Recall', f"{live_recall:.4f}")
                    st.metric('F1 Score', f"{live_f1:.4f}")
                with c3:
                    auc_display = f"{live_auc:.4f}" if live_auc != 'N/A' else 'N/A'
                    mcc_display = f"{live_mcc:.4f}" if live_mcc != 'N/A' else 'N/A'
                    st.metric('AUC', auc_display)
                    st.metric('MCC', mcc_display)
            else:
                st.error(f" Cannot calculate metrics: {alignment_msg}")
        else:
            st.warning('Could not prepare data for evaluation')
    except Exception as e:
        st.error(f'Error calculating metrics: {e}')

    # 4) Confusion matrix
    st.header('Confusion Matrix')
    
    try:
        X_test, y_test = prepare_data_for_prediction(test_df, target_col, scaler)
        if X_test is None:
            st.warning('Could not prepare test data')
            return
        
        X_aligned, alignment_msg, can_predict = align_features(X_test, model, metadata, 'pad_truncate')
        
        if can_predict:
            # Make prediction
            y_pred = model.predict(X_aligned)
        else:
            st.error(f"Cannot proceed with prediction: {alignment_msg}")
            return
            
        # prepare labels
        y_true = pd.Series(y_test).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)
        # binary fallback
        if y_true.nunique() > 2:
            top = sorted(y_true.unique())[-1]
            y_true_bin = (y_true == top).astype(int)
            y_pred_bin = (y_pred_s == top).astype(int)
        else:
            y_true_bin = y_true.astype(int)
            y_pred_bin = y_pred_s.astype(int)

        cm = confusion_matrix(y_true_bin, y_pred_bin)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    except Exception as e:
        st.error(f'Error during live evaluation: {e}')


if __name__ == '__main__':
    main()
