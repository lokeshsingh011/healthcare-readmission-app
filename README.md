
# 🏥 Healthcare Readmission Risk Prediction App

A Streamlit-based machine learning web app that predicts the risk of hospital readmission for diabetic patients using clinical data. Powered by **XGBoost**, with explainable AI support via **SHAP**.

---

## 🚀 Live Demo

🔗 [Click here to try the app](https://healthcare-readmission-app-gvxzcx9gkqy5q9wuhcfk6n.streamlit.app/)  

---

## 📌 Project Highlights

- ✅ Predicts **readmission risk** based on key patient features.
- ⚙️ Built using **XGBoost Classifier** with class imbalance handling.
- 📊 SHAP-based **interpretability** to visualize feature contributions.
- 🧠 Designed for **clinical insights** and patient care prioritization.
- 🌐 Deployable instantly via **Streamlit Cloud**.

---

## 🧪 Features Used for Prediction

- Number of Inpatient Visits
- Number of Diagnoses
- Time Spent in Hospital
- Diabetes Medication Prescribed (Yes/No)
- Number of Medications Prescribed
- Discharge Disposition ID
- One-hot encoded clinical history fields (e.g., Age Range, A1Cresult, Max Glucose Serum, etc.)

---

## 🧰 Tech Stack

| Component         | Technology          |
|------------------|---------------------|
| Language         | Python 3.10+        |
| Web App Framework| Streamlit           |
| Model            | XGBoost             |
| Preprocessing    | scikit-learn        |
| Explainability   | SHAP                |
| Visualization    | Matplotlib, SHAP    |

---

## 📂 Project Structure

```
healthcare_readmission_project/
├── app/
│   └── app.py                 # Streamlit frontend code
├── models/
│   ├── diabetes_model.pkl     # Trained model (XGBoost)
│   └── scaler.pkl             # Scaler used in training
├── data/
│   └── processed/
│       ├── shap_background.csv      # Background data for SHAP explainer
│       └── template_input.csv       # Template with one-hot columns for input
├── notebooks/                # EDA and model building notebooks
├── requirements.txt
└── README.md
```

> ⚠️ Note: The full dataset used for model training is not included due to GitHub size limits.  
> 📦 You can access the original dataset from [UCI ML Repository – Diabetes 130-US hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).

---

## 🧠 Model Performance

| Metric           | Score (Test Set)     |
|------------------|----------------------|
| Accuracy         | ~67%                 |
| Recall (Class 1) | ~57%                 |
| Precision (Class 1) | ~18%             |
| ROC AUC Score    | ~0.67                |

> Optimized for **Recall** to minimize false negatives in identifying high-risk patients.

---

## 🛠 How to Run Locally

1. **Clone the Repo**
   ```bash
   git clone https://github.com/lokeshsingh011/healthcare-readmission-app.git
   cd healthcare-readmission-app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit App**
   ```bash
   streamlit run app/app.py
   ```

---

## 📈 Future Improvements

- Upload CSV for batch prediction
- Add more patient history parameters
- Enable patient risk segmentation dashboard
- Integrate with EHR APIs

---

## 👨‍💻 Author

**Lokesh Kumar**  
🔗 [LinkedIn](https://www.linkedin.com/in/lokesh-kumar-ab41a819a/)  
📧 lkmahaur111@gmail.com

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
