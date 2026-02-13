try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. Please install with: pip install streamlit")
    raise
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
from sklearn.datasets import load_breast_cancer


st.set_page_config(page_title="ML classification models", layout="centered")


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
    # basic conversions
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.mean(numeric_only=True))
    try:
        if scaler is not None:
            X = scaler.transform(X)
        else:
            X = X.values
    except Exception as e:
        # If scaling fails, return as numpy array but preserve shape info
        X = X.values
    return X, y

def validate_features(X, model, metadata=None):
    """Validate feature count matches model expectations"""
    if X is None:
        return False, "No data provided"
    
    n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0]) if X else 0
    
    # Try to get expected features from model
    expected_features = None
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
    elif metadata and 'n_features' in metadata:
        expected_features = metadata['n_features']
    
    if expected_features is not None and n_features != expected_features:
        return False, f"Feature mismatch: Your data has {n_features} features, but the model expects {expected_features} features."
    
    return True, "Features validated successfully"


def load_default_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def main():
    st.title("ML classification models")
    st.markdown("""This web app evaluates multiple trained classification models; upload test data, choose a model, and view metrics, reports, and confusion matrices.""")
    st.markdown("- Upload a CSV test dataset\n- Select a trained model from a dropdown\n- View evaluation metrics (Accuracy, Precision, Recall, F1, AUC, MCC if available)\n- See Confusion Matrix and Classification Report")
    
    # Add dataset requirements info
    with st.expander("📋 Dataset Requirements", expanded=False):
        st.markdown("""
        **Important**: The models were trained on the **Breast Cancer Wisconsin dataset** with **30 features**.
        
        Your uploaded CSV should have:
        - **30 features** (same as training data)
        - Feature names matching the breast cancer dataset
        - A target column named 'target' or similar
        
        **Features expected**: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension.
        
        💡 **Tip**: Use the default dataset (loads automatically) or check the sample_test_data.csv for the correct format.
        """)

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
                # fallback: try to let pandas infer the separator (engine='python')
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
                test_df = pd.read_csv(uploaded, sep=None, engine='python')

            # If the file parsed into a single column, try common alternatives
            if test_df.shape[1] == 1:
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
                # try tab-separated
                try:
                    test_df = pd.read_csv(uploaded, sep='\t')
                except Exception:
                    try:
                        uploaded.seek(0)
                    except Exception:
                        pass
                    # try whitespace-delimited
                    test_df = pd.read_csv(uploaded, delim_whitespace=True)

            st.success(f"Loaded {len(test_df)} rows")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        # detect target column and show helpful info
        detected = auto_detect_target_column(test_df)
        # normalize column names (strip whitespace)
        test_df.columns = [c.strip() for c in test_df.columns]
        # try to ensure detected matches normalized names
        if detected not in test_df.columns:
            lower_map = {c.lower(): c for c in test_df.columns}
            if detected and detected.lower() in lower_map:
                detected = lower_map[detected.lower()]
            else:
                detected = test_df.columns[-1]

        # show detected column and its unique values for guidance
        st.markdown(f"**Detected target column:** {detected}")
        try:
            uniques = test_df[detected].dropna().unique()
            st.markdown(f"**Unique values (preview):** {list(uniques)[:10]}")
        except Exception:
            st.markdown("**Unique values (preview):** could not determine")

        # if multi-class, offer a conversion to binary
        try:
            u = test_df[detected].dropna().unique()
            if len(u) > 2:
                if st.checkbox('Convert multi-class target to binary (0 = no disease, 1 = disease)', value=True):
                    try:
                        test_df[detected] = (pd.to_numeric(test_df[detected], errors='coerce') > 0).astype(int)
                        st.success('Converted target to binary')
                    except Exception:
                        st.error('Failed to convert target to binary')
        except Exception:
            pass

        target_col = st.selectbox("Select target column", options=list(test_df.columns), index=list(test_df.columns).index(detected) if detected in test_df.columns else 0)
    else:
        test_df = load_default_dataset()
        target_col = 'target'

    # 2) Model selection
    available = [k for k,v in models.items() if v is not None]
    if not available:
        st.error('No trained models found. Place <name>_model.pkl files in the app folder.')
        return
    display = [k.replace('_',' ').title() for k in available]
    sel = st.selectbox('Select model', options=display)
    sel_key = available[display.index(sel)]
    model = models[sel_key]

    # 3) Display evaluation metrics (precomputed if available)
    st.header('Evaluation Metrics')
    if results_df is not None:
        row = results_df[results_df['Model'].str.contains(sel_key.replace('_',' '), case=False)]
        if len(row) > 0:
            r = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric('Accuracy', f"{r.get('Accuracy',0):.4f}")
                st.metric('Precision', f"{r.get('Precision',0):.4f}")
            with c2:
                st.metric('Recall', f"{r.get('Recall',0):.4f}")
                st.metric('F1 Score', f"{r.get('F1 Score',0):.4f}")
            with c3:
                st.metric('AUC', f"{r.get('AUC Score','N/A')}")
                st.metric('MCC', f"{r.get('MCC','N/A')}")
        else:
            st.info('No precomputed metrics for selected model')
    else:
        st.info('No model_results.csv found')

    # 4) Confusion matrix + classification report (live)
    st.header('Confusion Matrix & Classification Report')
    try:
        X_test, y_test = prepare_data_for_prediction(test_df, target_col, scaler)
        if X_test is None:
            st.warning('Could not prepare test data')
            return
        
        # Validate features before prediction
        is_valid, validation_msg = validate_features(X_test, model, metadata)
        if not is_valid:
            st.error(f"❌ {validation_msg}")
            st.info("💡 **Solution**: Please ensure your dataset has the same features as the training data. The models were trained on the Breast Cancer Wisconsin dataset with 30 features.")
            
            # Show expected vs actual features
            expected_features = getattr(model, 'n_features_in_', 'Unknown')
            actual_features = X_test.shape[1] if hasattr(X_test, 'shape') else 'Unknown'
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Features", expected_features)
            with col2:
                st.metric("Your Data Features", actual_features)
                
            st.info("Use the default dataset (breast cancer) or upload a CSV with the same 30 features as the training data.")
            return
            
        st.success(f"✅ Feature validation passed: {X_test.shape[1]} features")
        y_pred = model.predict(X_test)
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

        # classification report as plain text
        cr_text = classification_report(y_true_bin, y_pred_bin)
        st.text(cr_text)

        # AUC and MCC displayed as metrics
        auc_val = 'N/A'
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:,1]
            elif hasattr(model, 'decision_function'):
                probs = model.decision_function(X_test)
            else:
                probs = None
            if probs is not None and len(set(y_true_bin))==2:
                auc_val = roc_auc_score(y_true_bin, probs)
        except Exception:
            auc_val = 'N/A'
        try:
            mcc_val = matthews_corrcoef(y_true_bin, y_pred_bin)
        except Exception:
            mcc_val = 'N/A'
        m1, m2 = st.columns(2)
        with m1:
            st.metric('AUC', f"{auc_val}")
        with m2:
            st.metric('MCC', f"{mcc_val}")

    except Exception as e:
        st.error(f'Error during live evaluation: {e}')


if __name__ == '__main__':
    main()
