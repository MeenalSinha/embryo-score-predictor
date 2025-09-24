import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Page configuration
st.set_page_config(
    page_title="IVF Embryo Score Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 0rem;
    }
    .prediction-container {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-score {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-quality {
        font-size: 1.2rem;
        color: #1B5E20;
        font-weight: 500;
        text-align: center;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .quality-excellent { color: #2E7D32; font-weight: bold; }
    .quality-good { color: #689F38; font-weight: bold; }
    .quality-fair { color: #F57C00; font-weight: bold; }
    .quality-poor { color: #D32F2F; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Generate synthetic training data
@st.cache_data
def generate_training_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'day3_cells': np.random.randint(4, 17, n_samples),
        'day3_fragmentation': np.random.uniform(0, 50, n_samples),
        'day5_expansion': np.random.randint(1, 6, n_samples),
        'day5_icm_grade': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2]),
        'day5_te_grade': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2]),
        'maternal_age': np.random.uniform(18, 50, n_samples),
        'fertilization_day': np.random.choice([0, 1], n_samples),
        'culture_medium': np.random.choice([0, 1, 2], n_samples),
        'incubation_temp': np.random.uniform(36.5, 37.5, n_samples),
        'co2_concentration': np.random.uniform(5, 7, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create quality score based on parameters
    quality_score = (
        (df['day3_cells'] - 4) / 12 * 2 +
        (50 - df['day3_fragmentation']) / 50 * 2 +
        (df['day5_expansion'] - 1) / 4 * 2 +
        (4 - df['day5_icm_grade']) / 2 * 1.5 +
        (4 - df['day5_te_grade']) / 2 * 1.5 +
        (50 - df['maternal_age']) / 32 * 1 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    df['quality_score'] = np.clip(quality_score, 0, 10)
    
    return df

# Train the model
@st.cache_resource
def train_model():
    df = generate_training_data()
    
    X = df.drop('quality_score', axis=1)
    y = df['quality_score']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist()

# Load model
model, scaler, feature_names = train_model()

# Helper functions
def get_quality_interpretation(score):
    if score >= 8:
        return "Excellent Quality", "quality-excellent"
    elif score >= 6:
        return "Good Quality", "quality-good"
    elif score >= 4:
        return "Fair Quality", "quality-fair"
    else:
        return "Poor Quality", "quality-poor"

def create_feature_importance_chart():
    df = generate_training_data()
    X = df.drop('quality_score', axis=1)
    y = df['quality_score']
    X_scaled = scaler.transform(X)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': [
            'Day 3 Cells', 'Day 3 Fragmentation', 'Day 5 Expansion',
            'ICM Grade', 'TE Grade', 'Maternal Age',
            'Fertilization Day', 'Culture Medium', 'Incubation Temp', 'CO2 Concentration'
        ],
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        feature_importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Greens'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def create_score_distribution_chart():
    df = generate_training_data()
    
    fig = px.histogram(
        df, 
        x='quality_score', 
        nbins=20,
        color_discrete_sequence=['#4CAF50']
    )
    fig.update_layout(
        xaxis_title="Quality Score",
        yaxis_title="Frequency",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def create_correlation_heatmap():
    df = generate_training_data()
    correlation_matrix = df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 IVF Embryo Quality Score Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("📊 Embryo Parameters")
    
    # Day 3 Parameters
    st.sidebar.subheader("Day 3 Assessment")
    day3_cells = st.sidebar.slider("Cell Count", 4, 16, 8, help="Number of cells on day 3")
    day3_fragmentation = st.sidebar.slider("Fragmentation (%)", 0, 50, 10, help="Percentage of fragmentation")
    
    # Day 5 Parameters
    st.sidebar.subheader("Day 5 Assessment")
    day5_expansion = st.sidebar.selectbox("Expansion Grade", [1, 2, 3, 4, 5], index=2, help="Blastocyst expansion grade")
    day5_icm_grade = st.sidebar.selectbox("ICM Grade", ["A", "B", "C"], index=1, help="Inner Cell Mass grade")
    day5_te_grade = st.sidebar.selectbox("TE Grade", ["A", "B", "C"], index=1, help="Trophectoderm grade")
    
    # Clinical Parameters
    st.sidebar.subheader("Clinical Parameters")
    maternal_age = st.sidebar.slider("Maternal Age", 18, 50, 32, help="Age of the mother")
    fertilization_day = st.sidebar.selectbox("Fertilization Day", [0, 1], help="Day of fertilization")
    culture_medium = st.sidebar.selectbox("Culture Medium", ["Medium A", "Medium B", "Medium C"], help="Type of culture medium")
    incubation_temp = st.sidebar.slider("Incubation Temperature (°C)", 36.5, 37.5, 37.0, step=0.1)
    co2_concentration = st.sidebar.slider("CO2 Concentration (%)", 5.0, 7.0, 6.0, step=0.1)
    
    # Convert categorical inputs
    icm_mapping = {"A": 1, "B": 2, "C": 3}
    te_mapping = {"A": 1, "B": 2, "C": 3}
    medium_mapping = {"Medium A": 0, "Medium B": 1, "Medium C": 2}
    
    # Prepare input data
    input_data = np.array([[
        day3_cells,
        day3_fragmentation,
        day5_expansion,
        icm_mapping[day5_icm_grade],
        te_mapping[day5_te_grade],
        maternal_age,
        fertilization_day,
        medium_mapping[culture_medium],
        incubation_temp,
        co2_concentration
    ]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    quality_text, quality_class = get_quality_interpretation(prediction)
    
    # Display prediction
    st.markdown(f"""
    <div class="prediction-container">
        <div class="prediction-title">Predicted Embryo Quality Score 🔗</div>
        <div class="prediction-score">{prediction:.2f}/10</div>
        <div class="prediction-quality">{quality_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "🎯 Model Performance", "📋 Parameter Guide", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Importance")
            fig_importance = create_feature_importance_chart()
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("Score Distribution")
            fig_distribution = create_score_distribution_chart()
            st.plotly_chart(fig_distribution, use_container_width=True)
        
        st.subheader("Parameter Correlations")
        fig_correlation = create_correlation_heatmap()
        st.plotly_chart(fig_correlation, use_container_width=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        # Calculate model performance metrics
        df = generate_training_data()
        X = df.drop('quality_score', axis=1)
        y = df['quality_score']
        X_scaled = scaler.transform(X)
        
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R² Score</h3>
                <h2 style="color: #4CAF50;">{cv_scores.mean():.3f}</h2>
                <p>Cross-validated accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            predictions = model.predict(X_scaled)
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <h2 style="color: #4CAF50;">{rmse:.3f}</h2>
                <p>Root Mean Square Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Training Samples</h3>
                <h2 style="color: #4CAF50;">1,000</h2>
                <p>Synthetic embryo data</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Model Validation")
        st.write(f"**Cross-validation scores:** {cv_scores}")
        st.write(f"**Mean CV Score:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    with tab3:
        st.subheader("Parameter Interpretation Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Day 3 Parameters:**
            - **Cell Count**: 6-8 cells optimal on day 3
            - **Fragmentation**: <10% excellent, 10-25% acceptable
            
            **Day 5 Parameters:**
            - **Expansion**: 3-4 indicates good development
            - **ICM Grade**: A (excellent) > B (good) > C (poor)
            - **TE Grade**: A (excellent) > B (good) > C (poor)
            """)
        
        with col2:
            st.markdown("""
            **Clinical Parameters:**
            - **Maternal Age**: <35 years generally better outcomes
            - **Fertilization Day**: Day 0 or 1 post-retrieval
            - **Culture Medium**: Different formulations available
            - **Temperature**: 37°C ± 0.5°C optimal
            - **CO2**: 6% ± 1% standard concentration
            """)
        
        st.subheader("Quality Score Ranges")
        st.markdown("""
        - **8.0-10.0**: 🟢 Excellent quality - High implantation potential
        - **6.0-7.9**: 🔵 Good quality - Suitable for transfer
        - **4.0-5.9**: 🟡 Fair quality - Consider individual factors
        - **0.0-3.9**: 🔴 Poor quality - Additional assessment needed
        """)
    
    with tab4:
        st.subheader("About This Application")
        st.markdown("""
        This IVF Embryo Quality Score Predictor uses machine learning to estimate embryo quality based on 
        morphological and clinical parameters. The model is trained on synthetic data that reflects 
        real-world embryological patterns and relationships.
        
        **Key Features:**
        - Random Forest algorithm for robust predictions
        - Comprehensive parameter analysis
        - Interactive visualizations
        - Model performance metrics
        
        **Important Notes:**
        - This tool is for educational and research purposes only
        - Should not replace professional medical judgment
        - Always consult with fertility specialists for treatment decisions
        - Model trained on synthetic data for demonstration purposes
        
        **Technical Details:**
        - Algorithm: Random Forest Regressor (100 estimators)
        - Features: 10 embryological and clinical parameters
        - Training: 1,000 synthetic samples with realistic parameter relationships
        - Validation: 5-fold cross-validation
        """)
        
        st.subheader("Disclaimer")
        st.warning("""
        ⚠️ **Medical Disclaimer**: This application is for educational purposes only and should not be used 
        for actual clinical decision-making. Always consult with qualified healthcare professionals for 
        medical advice and treatment decisions.
        """)

if __name__ == "__main__":
    main()