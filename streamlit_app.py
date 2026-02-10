

def check_and_install_packages():
    """Check if required packages are installed and install if needed"""
    try:
        import streamlit as st
        return True, st
    except ImportError:
        print("❌ Streamlit not installed!")
        print("📦 Please install required packages with:")
        print("pip install streamlit plotly xgboost scikit-learn")
        return False, None

# Check packages and get streamlit
PACKAGES_OK, st = check_and_install_packages()

if not PACKAGES_OK:
    # If packages aren't ready, stop execution
    import sys
    print("🛑 Please install packages and restart before continuing.")
else:
    # Continue with imports only if streamlit is available
    import pandas as pd
    import numpy as np
    import joblib
    import pickle
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import warnings
    warnings.filterwarnings('ignore')

    # Page configuration
    st.set_page_config(
        page_title="ML Classification Models Comparison",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ff7f0e;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
            margin-bottom: 1rem;
        }
        .student-info {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #1f77b4;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title and Student Information
    st.markdown('<div class="main-header">🤖 Machine Learning Classification Models</div>', unsafe_allow_html=True)


    st.markdown('''
    <div class="student-info">
    <h4>📚 Assignment Information</h4>
    <strong>Student:</strong> Bhavani Mallem<br>
    <strong>BITS ID:</strong> 2025AB05021<br>
    <strong>Email:</strong> 2025ab05021@wilp.bits-pilani.ac.in<br>
    <strong>Date:</strong> February 6, 2026<br>
    <strong>Course:</strong> Machine Learning - Assignment 2
    </div>
    ''', unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        """Load the dataset and results"""
        try:
            # Load results
            results_df = pd.read_csv('model_results.csv')
            st.success("✅ Model results loaded successfully!")
            
            # Load Breast Cancer dataset for prediction demo
            try:
                from sklearn.datasets import load_breast_cancer
                cancer_data = load_breast_cancer()
                df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
                df['target'] = cancer_data.target
                return results_df, df.head(100)  # Sample for demo
            except:
                # Create dummy data if dataset not available
                st.warning("⚠️ Using dummy data for demonstration")
                np.random.seed(42)
                dummy_data = pd.DataFrame({
                    'mean radius': np.random.normal(14, 4, 100),
                    'mean texture': np.random.normal(19, 4, 100),
                    'mean perimeter': np.random.normal(92, 24, 100),
                    'target': np.random.choice([0, 1], 100)
                })
                return None, dummy_data
        except FileNotFoundError:
            st.error("❌ Model results not found! Please run the Jupyter notebook first to train the models and generate results.")
            st.info("📝 **Instructions:**")
            st.code("""
1. Open and run ml_classification_models.ipynb
2. Execute all cells to train models and save results
3. Then refresh this Streamlit app
            """)
            return None, None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

    @st.cache_resource
    def load_models():
        """Load all trained models and preprocessors"""
        models = {}
        try:
            model_names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'xgboost']
            loaded_models = 0
            
            for name in model_names:
                try:
                    models[name] = joblib.load(f'{name}_model.pkl')
                    loaded_models += 1
                except FileNotFoundError:
                    models[name] = None
                    
            # Load preprocessors
            try:
                scaler = joblib.load('scaler.pkl')
                with open('metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                st.success(f"✅ Successfully loaded {loaded_models}/{len(model_names)} models")
            except FileNotFoundError:
                scaler = None
                metadata = {
                    'feature_names': ['mean radius', 'mean texture', 'mean perimeter'], 
                    'target_classes': ['Malignant', 'Benign'],
                    'dataset_info': {'name': 'Breast Cancer Wisconsin', 'n_features': 30}
                }
                if loaded_models == 0:
                    st.warning("⚠️ No trained models found! Please run the notebook first.")
            
            return models, scaler, metadata
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return {}, None, {}

    def main():
        # Load data and models
        results_df, sample_data = load_data()
        models, scaler, metadata = load_models()
        
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["📊 Model Comparison", "🔍 Dataset Overview", "🎯 Model Predictions", "📈 Performance Analysis"]
        )
        
        # Check if data is available
        if results_df is None and page == "📊 Model Comparison":
            st.error("❌ Model results not available")
            st.info("📝 To fix this issue:")
            st.code("""
    # In Jupyter notebook or Colab:
    1. Run all cells in ml_classification_models.ipynb
    2. Ensure these files are created:
       - model_results.csv
       - *.pkl model files
       - scaler.pkl
       - metadata.pkl
    3. Refresh this Streamlit app
            """)
            return
        elif sample_data is None and page == "🔍 Dataset Overview":
            st.warning("⚠️ Using demo data - run notebook for full dataset")
        
        if page == "📊 Model Comparison":
            show_model_comparison(results_df)
        elif page == "🔍 Dataset Overview":
            show_dataset_overview(sample_data)
        elif page == "🎯 Model Predictions":
            show_predictions(models, scaler, metadata)
        elif page == "📈 Performance Analysis":
            show_performance_analysis(results_df)

    def show_model_comparison(results_df):
        st.markdown('<div class="sub-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
        
        # Display results table
        st.subheader("📋 Complete Results Table")
        st.dataframe(results_df, use_container_width=True)
        
        # Create interactive plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig_acc = px.bar(
                results_df, x='Model', y='Accuracy',
                title='Model Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # F1 Score comparison
            fig_f1 = px.bar(
                results_df, x='Model', y='F1 Score',
                title='Model F1 Score Comparison',
                color='F1 Score',
                color_continuous_scale='plasma'
            )
            fig_f1.update_layout(showlegend=False)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Radar chart for all metrics
        st.subheader("🎯 All Metrics Radar Chart")
        
        # Prepare data for radar chart (excluding AUC if it has N/A values)
        radar_data = results_df.copy()
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Check if AUC Score is available and numeric
        if 'AUC Score' in radar_data.columns:
            auc_numeric = pd.to_numeric(radar_data['AUC Score'], errors='coerce')
            if not auc_numeric.isna().all():
                metrics_to_plot.append('AUC Score')
                radar_data['AUC Score'] = auc_numeric
        
        fig_radar = go.Figure()
        
        for idx, row in radar_data.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics_to_plot],
                theta=metrics_to_plot,
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Best performers
        st.subheader("🏆 Best Performing Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
            st.markdown(f'''
            <div class="metric-card">
            <h4>🎯 Best Accuracy</h4>
            <strong>{best_accuracy["Model"]}</strong><br>
            Accuracy: {best_accuracy["Accuracy"]:.4f}
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            best_f1 = results_df.loc[results_df['F1 Score'].idxmax()]
            st.markdown(f'''
            <div class="metric-card">
            <h4>⚖️ Best F1 Score</h4>
            <strong>{best_f1["Model"]}</strong><br>
            F1 Score: {best_f1["F1 Score"]:.4f}
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            best_precision = results_df.loc[results_df['Precision'].idxmax()]
            st.markdown(f'''
            <div class="metric-card">
            <h4>🎪 Best Precision</h4>
            <strong>{best_precision["Model"]}</strong><br>
            Precision: {best_precision["Precision"]:.4f}
            </div>
            ''', unsafe_allow_html=True)

    def show_dataset_overview(sample_data):
        st.markdown('<div class="sub-header">🔍 Dataset Overview</div>', unsafe_allow_html=True)
        
        if sample_data is None:
            st.error("Dataset not available for overview.")
            return
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", f"{len(sample_data):,}")
        with col2:
            st.metric("Features", f"{len(sample_data.columns)-1}")
        with col3:
            st.metric("Classes", "2 (Binary)")
        with col4:
            st.metric("Problem Type", "Classification")
        
        # Display sample data
        st.subheader("📋 Sample Data")
        st.dataframe(sample_data.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("📊 Data Statistics")
        st.dataframe(sample_data.describe(), use_container_width=True)
        
        # Data visualization
        if 'diabetes' in sample_data.columns:
            st.subheader("📈 Target Distribution")
            
            target_counts = sample_data['diabetes'].value_counts()
            fig_target = px.pie(
                values=target_counts.values,
                names=['No Diabetes', 'Diabetes'],
                title='Target Class Distribution'
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
            # Feature distributions
            numeric_columns = sample_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                st.subheader("📊 Feature Distributions")
                
                selected_feature = st.selectbox("Select a feature to visualize:", numeric_columns)
                
                if selected_feature:
                    fig_dist = px.histogram(
                        sample_data, x=selected_feature, color='diabetes',
                        title=f'Distribution of {selected_feature} by Target',
                        nbins=30
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

    def show_predictions(models, scaler, metadata):
        st.markdown('<div class="sub-header">🎯 Model Predictions</div>', unsafe_allow_html=True)
        
        if not models or not metadata:
            st.error("Models not available for predictions.")
            return
        
        st.write("Enter feature values to get predictions from all models:")
        
        # Create input form based on available features
        feature_names = metadata.get('feature_names', ['feature1', 'feature2'])
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        inputs = {}
        
        # Create sample input values
        with col1:
            st.subheader("Patient Information")
            inputs['age'] = st.slider("Age", 18, 100, 45)
            inputs['bmi'] = st.slider("BMI", 15.0, 50.0, 25.0)
            inputs['glucose_level'] = st.slider("Glucose Level", 50, 300, 120)
        
        with col2:
            st.subheader("Additional Features")
            inputs['smoking_history'] = st.selectbox("Smoking History", [0, 1, 2], format_func=lambda x: ['Never', 'Former', 'Current'][x])
            inputs['heart_disease'] = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
            inputs['hypertension'] = st.selectbox("Hypertension", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
        
        if st.button("🔮 Make Predictions", type="primary"):
            # Create input array
            input_data = np.array([[inputs.get(f, 0) for f in feature_names[:len(inputs)]]])
            
            # Make predictions with each model
            predictions = {}
            probabilities = {}
            
            for name, model in models.items():
                if model is not None:
                    try:
                        # Use scaled data for models that need it
                        if name in ['logistic_regression', 'knn', 'naive_bayes'] and scaler is not None:
                            scaled_input = scaler.transform(input_data)
                            pred = model.predict(scaled_input)[0]
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(scaled_input)[0][1]
                            else:
                                prob = None
                        else:
                            pred = model.predict(input_data)[0]
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(input_data)[0][1]
                            else:
                                prob = None
                        
                        predictions[name] = pred
                        probabilities[name] = prob
                    except Exception as e:
                        predictions[name] = "Error"
                        probabilities[name] = None
            
            # Display predictions
            st.subheader("🎯 Prediction Results")
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.write("**Predictions:**")
            for name, pred in predictions.items():
                if pred != "Error":
                    result = "Diabetes" if pred == 1 else "No Diabetes"
                    color = "🔴" if pred == 1 else "🟢"
                    st.write(f"{color} **{name.replace('_', ' ').title()}**: {result}")
                else:
                    st.write(f"❌ **{name.replace('_', ' ').title()}**: Error")
        
        with pred_col2:
            st.write("**Confidence Scores:**")
            valid_probs = {k: v for k, v in probabilities.items() if v is not None}
            if valid_probs:
                prob_df = pd.DataFrame.from_dict(valid_probs, orient='index', columns=['Probability'])
                prob_df['Model'] = prob_df.index
                prob_df['Model'] = prob_df['Model'].str.replace('_', ' ').str.title()
                
                fig_prob = px.bar(
                    prob_df, x='Model', y='Probability',
                    title='Prediction Confidence (Diabetes Probability)',
                    color='Probability',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)

    def show_performance_analysis(results_df):
        st.markdown('<div class="sub-header">📈 Performance Analysis</div>', unsafe_allow_html=True)
        
        # Model ranking
        st.subheader("🏆 Model Ranking Analysis")
        
        # Calculate average rank across all metrics
        rank_df = results_df.copy()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Add AUC Score if available and numeric
        if 'AUC Score' in rank_df.columns:
            auc_numeric = pd.to_numeric(rank_df['AUC Score'], errors='coerce')
            if not auc_numeric.isna().all():
                metrics.append('AUC Score')
                rank_df['AUC Score'] = auc_numeric
        
        # Calculate ranks (1 = best)
        for metric in metrics:
            rank_df[f'{metric}_rank'] = rank_df[metric].rank(ascending=False)
        
        # Calculate average rank
        rank_columns = [f'{metric}_rank' for metric in metrics]
        rank_df['Average_Rank'] = rank_df[rank_columns].mean(axis=1)
        rank_df['Overall_Rank'] = rank_df['Average_Rank'].rank()
        
        # Display ranking table
        ranking_display = rank_df[['Model'] + metrics + ['Average_Rank', 'Overall_Rank']].sort_values('Overall_Rank')
        st.dataframe(ranking_display, use_container_width=True)
        
        # Strengths and weaknesses analysis
        st.subheader("💪 Model Strengths & Weaknesses")
        
        for idx, row in results_df.iterrows():
            model_name = row['Model']
            
            with st.expander(f"📊 {model_name} Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strengths:**")
                    strengths = []
                    if row['Accuracy'] >= results_df['Accuracy'].quantile(0.75):
                        strengths.append("High Accuracy")
                    if row['Precision'] >= results_df['Precision'].quantile(0.75):
                        strengths.append("High Precision")
                    if row['Recall'] >= results_df['Recall'].quantile(0.75):
                        strengths.append("High Recall")
                    if row['F1 Score'] >= results_df['F1 Score'].quantile(0.75):
                        strengths.append("High F1 Score")
                    
                    for strength in strengths:
                        st.write(f"✅ {strength}")
                    if not strengths:
                        st.write("⚠️ No standout strengths")
                with col2:
                    st.write("**Areas for Improvement:**")
                    weaknesses = []
                    if row['Accuracy'] <= results_df['Accuracy'].quantile(0.25):
                        weaknesses.append("Low Accuracy")
                    if row['Precision'] <= results_df['Precision'].quantile(0.25):
                        weaknesses.append("Low Precision")
                    if row['Recall'] <= results_df['Recall'].quantile(0.25):
                        weaknesses.append("Low Recall")
                    if row['F1 Score'] <= results_df['F1 Score'].quantile(0.25):
                        weaknesses.append("Low F1 Score")
                    
                    for weakness in weaknesses:
                        st.write(f"🔴 {weakness}")
                    if not weaknesses:
                        st.write("✨ Well-balanced performance")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        best_overall = ranking_display.iloc[0]
        best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
        best_f1 = results_df.loc[results_df['F1 Score'].idxmax()]
        
        st.info(f"""
        **📊 Analysis Summary:**
        
        🏆 **Best Overall Model**: {best_overall['Model']} (Average Rank: {best_overall['Average_Rank']:.2f})
        
        🎯 **Best for Accuracy**: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})
        
        ⚖️ **Best for Balanced Performance**: {best_f1['Model']} (F1: {best_f1['F1 Score']:.4f})
        
        **Recommendations:**
        - For general use: Choose {best_overall['Model']} for consistent performance
        - For high accuracy needs: Use {best_accuracy['Model']}
        - For balanced precision/recall: Consider {best_f1['Model']}
        """)

    if __name__ == "__main__":
        main()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; margin-top: 2rem;'>
        <p>🎓 Machine Learning Assignment 2 | BITS Pilani | 2025AB05021 - Bhavani Mallem</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
