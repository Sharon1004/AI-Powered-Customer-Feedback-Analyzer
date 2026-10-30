# 💬 AI-Powered Customer Feedback Analyzer

An AI-powered application that analyzes customer feedback and predicts **Sentiment, Complaint Category, and Priority** using Natural Language Processing (NLP), Machine Learning, and Artificial Neural Networks (ANN).

## 🚀 Live Demo

**Try the application here:**

[Open AI-Powered Customer Feedback Analyzer](https://ai-powered-customer-feedback-analyzer.streamlit.app/)

## 📌 Project Overview

Understanding customer feedback is important for improving products, services, and customer satisfaction.

This project analyzes customer feedback and automatically predicts:

- **Sentiment Analysis:** Positive, Negative, or Neutral.
- **Complaint Classification:** Identifies the type of complaint.
- **Priority Prediction:** Classifies feedback as Low, Medium, or High priority.

The application provides an interactive Streamlit interface where users can enter customer feedback and receive predictions.

## ✨ Features

- Customer feedback analysis.
- Sentiment classification.
- Automatic complaint category detection.
- Priority prediction using an Artificial Neural Network.
- TF-IDF text feature extraction.
- Machine Learning classification using Logistic Regression.
- Model evaluation using accuracy, precision, recall, and F1-score.
- Confusion matrix visualizations.
- Interactive Streamlit web application.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and NLP |
| TF-IDF | Text feature extraction |
| Logistic Regression | Sentiment and complaint classification |
| TensorFlow / Keras | Artificial Neural Network |
| Matplotlib | Data visualization |
| Streamlit | Web application |

## 🧠 Machine Learning Approach

### 1. Sentiment Analysis
- Text preprocessing and cleaning.
- TF-IDF vectorization.
- Logistic Regression classification.
- Predicts Positive, Negative, or Neutral sentiment.

### 2. Complaint Classification
- TF-IDF feature extraction.
- Logistic Regression classification.
- Predicts the complaint category.

### 3. Priority Prediction
- TF-IDF text features.
- Customer metadata preprocessing.
- One-hot encoding for categorical features.
- Standard scaling for numerical features.
- Artificial Neural Network built using TensorFlow/Keras.
- Predicts Low, Medium, or High priority.

## 📂 Project Structure

```text
AI-Powered-Customer-Feedback-Analyzer/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
│
├── data/
│   └── customer_feedback.csv
│
├── models/
│   ├── sentiment_pipeline.pkl
│   ├── complaint_model.pkl
│   ├── priority_ann.keras
│   ├── priority_vectorizer.pkl
│   ├── priority_label_encoder.pkl
│   └── priority_preprocessor.pkl
│
├── notebooks/
│   └── project_workflow.ipynb
│
└── reports/
    ├── evaluation_metrics.json
    ├── evaluation_report.md
    ├── sentiment_confusion_matrix.png
    ├── complaint_type_confusion_matrix.png
    └── priority_confusion_matrix.png
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sharon1004/AI-Powered-Customer-Feedback-Analyzer.git
```

### 2. Navigate to the project directory

```bash
cd AI-Powered-Customer-Feedback-Analyzer
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the models

```bash
python train_models.py
```

This trains the Machine Learning and ANN models and saves the trained model files inside the `models/` directory.

### 7. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Evaluation results and confusion matrix visualizations are saved in the `reports/` directory after training.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live Application:** [AI-Powered Customer Feedback Analyzer](https://ai-powered-customer-feedback-analyzer.streamlit.app/)

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing.
- Text preprocessing and TF-IDF vectorization.
- Traditional Machine Learning classification.
- Artificial Neural Networks using TensorFlow/Keras.
- Model evaluation and performance analysis.
- Building interactive applications using Streamlit.
- Deploying Machine Learning applications.

## 👩‍💻 Author

**Sharon**

GitHub: [Sharon1004](https://github.com/Sharon1004)

---

⭐ If you find this project interesting, consider giving it a star!
