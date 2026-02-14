import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# Page configuration
st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💰 Adult Income Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict Income >50K or ≤50K using Census Data</p>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all saved models"""
    try:
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Decision Tree': joblib.load('models/decision_tree.pkl'),
            'K-Nearest Neighbors': joblib.load('models/knn.pkl'),
            'Naive Bayes': joblib.load('models/naive_bayes.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'XGBoost': joblib.load('models/xgboost.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Evaluation function
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
    results = {}
    results['Accuracy'] = accuracy_score(y_true, y_pred)
    results['Precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    results['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    results['F1 Score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    results['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    if y_pred_proba is not None:
        try:
            results['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            results['AUC'] = 'N/A'
    else:
        results['AUC'] = 'N/A'
    
    return results

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Load models
models, scaler = load_models()

if models is None:
    st.error("❌ Failed to load models")
    st.stop()
else:
    st.sidebar.success("✅ Models loaded!")

# Model selection
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    help="Choose a classification model"
)

# File upload
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="CSV with 'target' column"
)

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Samples", df.shape[0])
        with col2:
            st.metric("📈 Features", df.shape[1] - 1)
        with col3:
            st.metric("✅ Target", "Present" if 'target' in df.columns else "Missing")
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10))
        
        if 'target' not in df.columns:
            st.error("❌ 'target' column not found!")
            st.stop()
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Process features
        selected_model = models[model_choice]
        models_need_scaling = ['Logistic Regression', 'K-Nearest Neighbors']
        
        if model_choice in models_need_scaling:
            X_processed = scaler.transform(X)
        else:
            X_processed = X.values
        
        # Predictions
        with st.spinner('🔄 Making predictions...'):
            y_pred = selected_model.predict(X_processed)
            y_pred_proba = selected_model.predict_proba(X_processed)
        
        # Metrics
        results = evaluate_model(y, y_pred, y_pred_proba)
        
        st.markdown("---")
        st.markdown(f"### 📊 {model_choice} - Performance Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Accuracy", f"{results['Accuracy']:.4f}")
        with col2:
            auc_val = results['AUC'] if isinstance(results['AUC'], str) else f"{results['AUC']:.4f}"
            st.metric("AUC", auc_val)
        with col3:
            st.metric("Precision", f"{results['Precision']:.4f}")
        with col4:
            st.metric("Recall", f"{results['Recall']:.4f}")
        with col5:
            st.metric("F1 Score", f"{results['F1 Score']:.4f}")
        with col6:
            st.metric("MCC", f"{results['MCC']:.4f}")
        
        # Visualizations
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['≤50K', '>50K'],
                       yticklabels=['≤50K', '>50K'])
            ax.set_title(f'{model_choice}', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📈 ROC Curve")
            if y_pred_proba is not None and results['AUC'] != 'N/A':
                fig, ax = plt.subplots(figsize=(6, 5))
                fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
                ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {results["AUC"]:.4f}')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{model_choice}', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("ROC Curve not available")
        
        # Classification Report
        st.markdown("---")
        st.markdown("### 📋 Classification Report")
        report = classification_report(y, y_pred, 
                                      target_names=['≤50K', '>50K'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', 
                                                         subset=['precision', 'recall', 'f1-score']),
                    use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Upload CSV file to begin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 How to Use")
        st.markdown("""
        1. Select model from sidebar
        2. Upload CSV with 'target' column
        3. View comprehensive results
        4. Compare different models
        
        ### ✨ Features
        - ✅ 6 Classification Models
        - ✅ Comprehensive Metrics
        - ✅ Visual Analytics
        - ✅ Detailed Reports
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Info")
        st.markdown("""
        **Adult Income (UCI)**
        
        - **Instances:** 48,842
        - **Features:** 14
        - **Classes:** ≤50K vs >50K
        - **Task:** Binary Classification
        
        **Goal:** Predict if income > $50K
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>ML Assignment 2</strong> | TUSHAR KANTI SANTRA | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
