
# 🧬 IVF Embryo Quality Predictor

An AI-powered application that predicts **IVF embryo viability** using deep learning (CNN + Grad-CAM).  
Built with **Streamlit** for an interactive interface and deployed on **Streamlit Cloud**.

---

## 🔗 Live Demo
👉 [Embryo Score Predictor](https://embryo-score-predictor-d9wjkfw8dezzlmwclkbjph.streamlit.app/)  

---

## ✨ Features
- **Deep Learning Prediction**: CNN-based classifier for viable vs. non-viable embryos  
- **Explainability**: Grad-CAM heatmaps highlight regions influencing predictions  
- **Interactive Interface**: Upload embryo images and get instant quality scores  
- **Exportable Reports**: Generate PDF/CSV outputs for easy record-keeping  
- **Streamlit Cloud Deployment**: Accessible from any browser, no setup required  

---

## 📊 Dataset & Methodology
- **Dataset**: Open-source embryo images (Kaggle)  
- **Preprocessing**: Augmentation, normalization, and class-weighting to handle imbalance  
- **Model**: Custom CNN trained for binary classification (Viable vs Non-viable)  
- **Explainability**: Grad-CAM visualizations for clinical interpretability  
- **Frameworks**: PyTorch / TensorFlow (CNN), Streamlit (frontend), Matplotlib (visuals)  

---

## 📈 Model Performance
- **Accuracy**: ~87%  
- **Precision (Viable)**: 0.79  
- **Recall (Viable)**: 0.76  
- **Precision (Non-viable)**: 0.92  
- **Recall (Non-viable)**: 0.93  

📌 *Note*: Model was trained on limited open datasets; real-world deployment requires clinical validation.  

---

## 🖼 Demo Screenshots

| Upload Embryo | Grad-CAM Visualization | Export Report |
|---------------|------------------------|---------------|
| <img width="1483" alt="Upload Embryo" src="https://github.com/user-attachments/assets/e727e70d-601c-4c0d-a87f-a43fb160b726" /> | <img width="1812" alt="Grad-CAM Visualization" src="https://github.com/user-attachments/assets/b0c10633-c000-4704-963b-cd6e8523a275" /> | <img width="1770" alt="Export Report" src="https://github.com/user-attachments/assets/abe26e15-a644-41f1-8b18-bf4bcdd1a634" /> |  

---

## ⚠️ Disclaimer
This application is for **educational and research purposes only**.  
It must **not** replace professional medical judgment or clinical decision-making.  
Always consult with qualified fertility specialists for treatment decisions.  

---

## 🔧 Installation (Local Development)

```bash
# Clone the repository
git clone https://github.com/MeenalSinha/embryo-score-predictor.git
cd embryo-score-predictor

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
````

---

## 🛠 Tech Stack

* **Python 3.9+**
* **Deep Learning**: CNN + Grad-CAM
* **Frameworks**: PyTorch / TensorFlow, Streamlit, OpenCV, Matplotlib
* **Deployment**: Streamlit Cloud

---

## 🌍 AI for Social Good

This project is submitted under **Track 1: AI for Social Good (Healthcare)** at AIGNITION.

* 🌍 **Global Scalability**: Accessible IVF support worldwide
* ⚖️ **Democratizing IVF**: Reducing subjectivity and bias in embryo selection
* 👨‍👩‍👦 **Empowering Families**: Improving success rates, reducing financial and emotional burden

---

## 🤝 Contributing

Contributions are welcome!

* Model improvements (larger datasets, better CNNs)
* UI/UX enhancements for Streamlit app
* Integration with real clinical datasets

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* Fertility specialists for domain expertise
* Open-source machine learning community
* Streamlit team for the excellent framework
