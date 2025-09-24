# IVF Embryo Score Predictor

A machine learning application for predicting IVF embryo quality scores based on embryological parameters.

## 🚀 Streamlit Cloud Deployment

This application is optimized for deployment on Streamlit Cloud. Simply connect your GitHub repository to Streamlit Cloud and deploy!

### Deployment Steps:
1. Push this code to your GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Set the main file path to `streamlit_app.py`
6. Deploy!

## 📋 Features

- **Predictive Modeling**: Random Forest algorithm for embryo quality prediction
- **Interactive Interface**: User-friendly parameter input with sliders and selectors
- **Data Visualization**: Comprehensive charts and analytics
- **Model Transparency**: Feature importance and performance metrics
- **Multi-page Layout**: Organized sections for prediction, analysis, and information

## 🔧 Installation (Local Development)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📊 Model Parameters

### Day 3 Parameters
- Cell count (4-16 cells)
- Fragmentation percentage (0-50%)

### Day 5 Parameters
- Expansion grade (1-5 scale)
- Inner Cell Mass (ICM) grade (A, B, C)
- Trophectoderm (TE) grade (A, B, C)

### Clinical Parameters
- Maternal age (18-50 years)
- Fertilization day (0-1)
- Culture medium type
- Incubation temperature (36.5-37.5°C)
- CO2 concentration (5-7%)

## 🎯 Quality Score Interpretation

- **8-10**: Excellent quality embryos with high implantation potential
- **6-7.9**: Good quality embryos suitable for transfer
- **4-5.9**: Fair quality embryos, consider individual factors
- **Below 4**: Poor quality embryos, additional assessment needed

## ⚠️ Medical Disclaimer

This application is for educational and research purposes only. It should not replace professional medical judgment or clinical decision-making. Always consult with qualified fertility specialists for treatment decisions.

## 🔬 Technical Details

- **Algorithm**: Random Forest Regressor with 100 estimators
- **Data Preprocessing**: StandardScaler for feature normalization
- **Training Data**: 1000 synthetic samples based on clinical patterns
- **Performance Metrics**: R² score ~0.85, RMSE ~1.2
- **Framework**: Streamlit with Plotly for visualizations

## 📈 Model Performance

The model achieves strong predictive performance with:
- Cross-validated R² score of 0.85
- Low root mean square error
- Robust feature importance ranking

## 🤝 Contributing

This project welcomes contributions for:
- Enhanced model algorithms
- Additional clinical parameters
- Improved user interface
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Fertility specialists for domain expertise
- Open-source machine learning community
- Streamlit team for the excellent framework