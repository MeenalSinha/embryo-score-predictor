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
import base64

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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .prediction-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .prediction-score {
        color: #FFD700;
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-quality {
        color: white;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic training data
@st.cache_data
def generate_training_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'day3_cell_count': np.random.randint(4, 17, n_samples),
        'day3_fragmentation': np.random.uniform(0, 50, n_samples),
        'day5_expansion': np.random.randint(1, 6, n_samples),
        'day5_icm_grade': np.random.choice([3, 2, 1], n_samples, p=[0.3, 0.5, 0.2]),  # A=3, B=2, C=1
        'day5_te_grade': np.random.choice([3, 2, 1], n_samples, p=[0.25, 0.5, 0.25]),
        'maternal_age': np.random.uniform(18, 50, n_samples),
        'fertilization_day': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'culture_medium': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2]),  # Sequential=1, Single=2, Custom=3
        'incubation_temp': np.random.uniform(36.5, 37.5, n_samples),
        'co2_concentration': np.random.uniform(5.0, 7.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable (quality score) based on realistic relationships
    quality_score = (
        (df['day3_cell_count'] - 4) * 0.3 +  # Optimal around 8-12 cells
        (50 - df['day3_fragmentation']) * 0.08 +  # Less fragmentation is better
        df['day5_expansion'] * 0.8 +  # Higher expansion is better
        df['day5_icm_grade'] * 1.2 +  # A grade is best
        df['day5_te_grade'] * 1.0 +  # A grade is best
        (40 - np.abs(df['maternal_age'] - 30)) * 0.05 +  # Optimal around 30
        (1 - df['fertilization_day']) * 0.5 +  # Day 0 fertilization preferred
        (df['culture_medium'] == 1) * 0.3 +  # Sequential medium bonus
        (37.0 - np.abs(df['incubation_temp'] - 37.0)) * 2 +  # Optimal at 37°C
        (6.0 - np.abs(df['co2_concentration'] - 6.0)) * 0.5 +  # Optimal at 6%
        np.random.normal(0, 1, n_samples)  # Add some noise
    )
    
    # Normalize to 0-10 scale
    quality_score = np.clip(quality_score, 0, 10)
    df['quality_score'] = quality_score
    
    return df

# Train the model
@st.cache_resource
def train_model():
    df = generate_training_data()
    
    feature_columns = [
        'day3_cell_count', 'day3_fragmentation', 'day5_expansion',
        'day5_icm_grade', 'day5_te_grade', 'maternal_age',
        'fertilization_day', 'culture_medium', 'incubation_temp',
        'co2_concentration'
    ]
    
    X = df[feature_columns]
    y = df['quality_score']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    return model, scaler, feature_columns, cv_scores.mean(), df

# Load model and data
model, scaler, feature_columns, cv_score, training_data = train_model()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 IVF Embryo Quality Score Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Analysis", "Data Insights", "About"])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Model Analysis":
        model_analysis_page()
    elif page == "Data Insights":
        data_insights_page()
    else:
        about_page()

def prediction_page():
    st.markdown('<h2 class="sub-header">📊 Embryo Quality Prediction</h2>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Day 3 Parameters")
        day3_cell_count = st.slider("Cell Count", 4, 16, 8, help="Number of cells on day 3")
        day3_fragmentation = st.slider("Fragmentation (%)", 0, 50, 10, help="Percentage of fragmentation")
        
        st.markdown("### Day 5 Parameters")
        day5_expansion = st.slider("Expansion Grade", 1, 5, 3, help="Blastocyst expansion grade (1-5)")
        day5_icm_grade = st.selectbox("ICM Grade", ["A", "B", "C"], index=1, help="Inner Cell Mass grade")
        day5_te_grade = st.selectbox("TE Grade", ["A", "B", "C"], index=1, help="Trophectoderm grade")
    
    with col2:
        st.markdown("### Clinical Parameters")
        maternal_age = st.slider("Maternal Age", 18, 50, 30, help="Age of the mother")
        fertilization_day = st.selectbox("Fertilization Day", [0, 1], index=0, help="Day of fertilization")
        culture_medium = st.selectbox("Culture Medium", ["Sequential", "Single Step", "Custom"], index=0)
        
        st.markdown("### Laboratory Conditions")
        incubation_temp = st.slider("Incubation Temperature (°C)", 36.5, 37.5, 37.0, step=0.1)
        co2_concentration = st.slider("CO₂ Concentration (%)", 5.0, 7.0, 6.0, step=0.1)
    
    # Convert categorical variables to numerical
    icm_mapping = {"A": 3, "B": 2, "C": 1}
    te_mapping = {"A": 3, "B": 2, "C": 1}
    medium_mapping = {"Sequential": 1, "Single Step": 2, "Custom": 3}
    
    # Prepare input data
    input_data = np.array([[
        day3_cell_count,
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
    
    # Determine quality category
    if prediction >= 8:
        quality_text = "Excellent Quality - High implantation potential"
        quality_color = "#28a745"
    elif prediction >= 6:
        quality_text = "Good Quality - Suitable for transfer"
        quality_color = "#17a2b8"
    elif prediction >= 4:
        quality_text = "Fair Quality - Consider individual factors"
        quality_color = "#ffc107"
    else:
        quality_text = "Poor Quality - Additional assessment needed"
        quality_color = "#dc3545"
    
    # Display prediction
    st.markdown(f"""
    <div class="prediction-container">
        <div class="prediction-title">Predicted Embryo Quality Score 🔗</div>
        <div class="prediction-score">{prediction:.2f}/10</div>
        <div class="prediction-quality">{quality_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance for this prediction
    st.markdown("### 📈 Feature Importance for This Prediction")
    
    feature_importance = model.feature_importances_
    feature_names = [
        'Day 3 Cell Count', 'Day 3 Fragmentation', 'Day 5 Expansion',
        'ICM Grade', 'TE Grade', 'Maternal Age',
        'Fertilization Day', 'Culture Medium', 'Incubation Temp',
        'CO₂ Concentration'
    ]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance in Prediction Model",
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Confidence interval
    st.markdown("### 🎯 Prediction Confidence")
    
    # Generate multiple predictions with slight variations to estimate uncertainty
    n_bootstrap = 100
    bootstrap_predictions = []
    
    for _ in range(n_bootstrap):
        # Add small random noise to simulate uncertainty
        noise = np.random.normal(0, 0.1, input_scaled.shape)
        noisy_input = input_scaled + noise
        pred = model.predict(noisy_input)[0]
        bootstrap_predictions.append(pred)
    
    confidence_lower = np.percentile(bootstrap_predictions, 2.5)
    confidence_upper = np.percentile(bootstrap_predictions, 97.5)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower Bound (95% CI)", f"{confidence_lower:.2f}")
    with col2:
        st.metric("Predicted Score", f"{prediction:.2f}")
    with col3:
        st.metric("Upper Bound (95% CI)", f"{confidence_upper:.2f}")
    
    # Generate PDF report
    if st.button("📄 Generate PDF Report"):
        pdf_buffer = generate_pdf_report(input_data[0], prediction, quality_text, feature_names, feature_importance)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=f"embryo_quality_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

def model_analysis_page():
    st.markdown('<h2 class="sub-header">🔬 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Cross-Validation R² Score</h3>
            <h2 style="color: #1f77b4;">{cv_score:.3f}</h2>
            <p>Model explains {cv_score*100:.1f}% of variance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate RMSE
        y_pred = model.predict(scaler.transform(training_data[feature_columns]))
        rmse = np.sqrt(np.mean((training_data['quality_score'] - y_pred) ** 2))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Root Mean Square Error</h3>
            <h2 style="color: #17a2b8;">{rmse:.3f}</h2>
            <p>Average prediction error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate MAE
        mae = np.mean(np.abs(training_data['quality_score'] - y_pred))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Mean Absolute Error</h3>
            <h2 style="color: #28a745;">{mae:.3f}</h2>
            <p>Average absolute error</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction vs Actual scatter plot
    st.markdown("### 📊 Prediction vs Actual Values")
    
    fig_scatter = px.scatter(
        x=training_data['quality_score'],
        y=y_pred,
        title="Predicted vs Actual Quality Scores",
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        opacity=0.6
    )
    
    # Add perfect prediction line
    min_val = min(training_data['quality_score'].min(), y_pred.min())
    max_val = max(training_data['quality_score'].max(), y_pred.max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Residuals plot
    st.markdown("### 📈 Residuals Analysis")
    
    residuals = training_data['quality_score'] - y_pred
    
    fig_residuals = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residuals Distribution')
    )
    
    # Residuals vs predicted
    fig_residuals.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            opacity=0.6
        ),
        row=1, col=1
    )
    
    # Residuals histogram
    fig_residuals.add_trace(
        go.Histogram(
            x=residuals,
            name='Distribution',
            nbinsx=30
        ),
        row=1, col=2
    )
    
    fig_residuals.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_residuals, use_container_width=True)

def data_insights_page():
    st.markdown('<h2 class="sub-header">📊 Data Insights & Patterns</h2>', unsafe_allow_html=True)
    
    # Quality score distribution
    st.markdown("### Distribution of Quality Scores")
    
    fig_dist = px.histogram(
        training_data,
        x='quality_score',
        nbins=30,
        title="Distribution of Embryo Quality Scores",
        color_discrete_sequence=['#1f77b4']
    )
    fig_dist.update_layout(
        xaxis_title="Quality Score",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### Feature Correlations")
    
    correlation_matrix = training_data[feature_columns + ['quality_score']].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Quality by maternal age
    st.markdown("### Quality Score vs Maternal Age")
    
    # Create age bins
    training_data['age_group'] = pd.cut(
        training_data['maternal_age'],
        bins=[18, 25, 30, 35, 40, 50],
        labels=['18-25', '26-30', '31-35', '36-40', '41-50']
    )
    
    fig_age = px.box(
        training_data,
        x='age_group',
        y='quality_score',
        title="Quality Score Distribution by Maternal Age Group",
        color='age_group'
    )
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Day 3 vs Day 5 parameters
    st.markdown("### Day 3 vs Day 5 Parameter Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_day3 = px.scatter(
            training_data,
            x='day3_cell_count',
            y='quality_score',
            color='day3_fragmentation',
            title="Quality vs Day 3 Cell Count",
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_day3, use_container_width=True)
    
    with col2:
        fig_day5 = px.scatter(
            training_data,
            x='day5_expansion',
            y='quality_score',
            color='day5_icm_grade',
            title="Quality vs Day 5 Expansion",
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_day5, use_container_width=True)

def about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Purpose</h3>
        <p>This application uses machine learning to predict IVF embryo quality scores based on various embryological and clinical parameters. It's designed to assist fertility specialists in making informed decisions about embryo selection and transfer.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🔬 Methodology</h3>
        <ul>
            <li><strong>Algorithm:</strong> Random Forest Regressor with 100 estimators</li>
            <li><strong>Features:</strong> 10 key parameters including morphological and clinical factors</li>
            <li><strong>Training Data:</strong> 1000 synthetic samples based on clinical patterns</li>
            <li><strong>Validation:</strong> 5-fold cross-validation with R² score of {:.3f}</li>
        </ul>
    </div>
    """.format(cv_score), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📋 Parameters Explained</h3>
        <ul>
            <li><strong>Day 3 Cell Count:</strong> Number of cells in the embryo on day 3 (optimal: 8-12)</li>
            <li><strong>Day 3 Fragmentation:</strong> Percentage of cellular fragmentation (lower is better)</li>
            <li><strong>Day 5 Expansion:</strong> Blastocyst expansion grade (1-5 scale)</li>
            <li><strong>ICM Grade:</strong> Inner Cell Mass quality (A=best, B=good, C=fair)</li>
            <li><strong>TE Grade:</strong> Trophectoderm quality (A=best, B=good, C=fair)</li>
            <li><strong>Maternal Age:</strong> Age of the patient (optimal range: 25-35)</li>
            <li><strong>Fertilization Day:</strong> Day of fertilization (0 or 1)</li>
            <li><strong>Culture Medium:</strong> Type of culture medium used</li>
            <li><strong>Laboratory Conditions:</strong> Temperature and CO₂ concentration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Important Disclaimer</h3>
        <p><strong>This application is for educational and research purposes only.</strong> It should not replace professional medical judgment or clinical decision-making. Always consult with qualified fertility specialists for treatment decisions. The predictions are based on synthetic data and should be validated with real clinical data before any clinical application.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🏥 Quality Score Interpretation</h3>
        <ul>
            <li><strong>8-10:</strong> Excellent quality embryos with high implantation potential</li>
            <li><strong>6-7.9:</strong> Good quality embryos suitable for transfer</li>
            <li><strong>4-5.9:</strong> Fair quality embryos, consider individual factors</li>
            <li><strong>Below 4:</strong> Poor quality embryos, additional assessment needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical specifications
    st.markdown("### 🛠️ Technical Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Details:**
        - Algorithm: Random Forest
        - Estimators: 100
        - Max Depth: 10
        - Random State: 42
        """)
    
    with col2:
        st.markdown("""
        **Performance Metrics:**
        - R² Score: {:.3f}
        - RMSE: {:.3f}
        - Cross-validation: 5-fold
        """.format(cv_score, np.sqrt(np.mean((training_data['quality_score'] - model.predict(scaler.transform(training_data[feature_columns]))) ** 2))))

def generate_pdf_report(input_data, prediction, quality_text, feature_names, feature_importance):
    """Generate a PDF report of the prediction"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("IVF Embryo Quality Score Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Prediction result
    prediction_text = f"Predicted Quality Score: {prediction:.2f}/10"
    prediction_para = Paragraph(prediction_text, styles['Heading2'])
    story.append(prediction_para)
    
    quality_para = Paragraph(f"Quality Assessment: {quality_text}", styles['Normal'])
    story.append(quality_para)
    story.append(Spacer(1, 12))
    
    # Input parameters
    params_title = Paragraph("Input Parameters:", styles['Heading3'])
    story.append(params_title)
    
    param_text = f"""
    Day 3 Cell Count: {input_data[0]}<br/>
    Day 3 Fragmentation: {input_data[1]:.1f}%<br/>
    Day 5 Expansion: {input_data[2]}<br/>
    ICM Grade: {input_data[3]}<br/>
    TE Grade: {input_data[4]}<br/>
    Maternal Age: {input_data[5]:.0f} years<br/>
    Fertilization Day: {input_data[6]}<br/>
    Culture Medium: {input_data[7]}<br/>
    Incubation Temperature: {input_data[8]:.1f}°C<br/>
    CO₂ Concentration: {input_data[9]:.1f}%
    """
    
    params_para = Paragraph(param_text, styles['Normal'])
    story.append(params_para)
    story.append(Spacer(1, 12))
    
    # Feature importance
    importance_title = Paragraph("Feature Importance:", styles['Heading3'])
    story.append(importance_title)
    
    importance_text = ""
    for name, importance in zip(feature_names, feature_importance):
        importance_text += f"{name}: {importance:.3f}<br/>"
    
    importance_para = Paragraph(importance_text, styles['Normal'])
    story.append(importance_para)
    
    # Disclaimer
    story.append(Spacer(1, 12))
    disclaimer_title = Paragraph("Disclaimer:", styles['Heading3'])
    story.append(disclaimer_title)
    
    disclaimer_text = """
    This report is generated for educational and research purposes only. 
    It should not replace professional medical judgment or clinical decision-making. 
    Always consult with qualified fertility specialists for treatment decisions.
    """
    disclaimer_para = Paragraph(disclaimer_text, styles['Normal'])
    story.append(disclaimer_para)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main()