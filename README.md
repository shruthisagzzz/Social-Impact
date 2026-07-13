# 🤖 AI-Driven Framework for Automated Grant Eligibility Evaluation and Impact Prediction

> **An AI-powered decision support system that automates grant proposal evaluation using Natural Language Processing (NLP) and Machine Learning (ML), enabling faster, fairer, and more transparent funding decisions.**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-AI-green?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange?style=for-the-badge)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-red?style=for-the-badge&logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</p>

---

# 📖 Overview

Grant funding plays a vital role in supporting projects that generate social, economic, and environmental impact. However, traditional grant evaluation is often slow, subjective, and difficult to scale due to the large volume of proposals received by funding organizations.

The **AI-Driven Framework for Automated Grant Eligibility Evaluation and Impact Prediction** is designed to automate the grant evaluation process using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

The system analyzes proposal documents, extracts meaningful textual features, evaluates proposal relevance using **TF-IDF and Cosine Similarity**, predicts social impact, and recommends whether a proposal should be **Approved**, **Partially Funded**, or **Rejected**.

The platform serves as a decision-support system for government agencies, NGOs, CSR organizations, and private funding bodies by improving transparency, consistency, and efficiency in grant allocation.

---

# 🎯 Problem Statement

Traditional grant evaluation suffers from several challenges:

- Manual review of thousands of proposals
- Inconsistent evaluation criteria
- Human bias in decision making
- Long processing time
- Lack of transparency
- Difficulty in identifying high-impact proposals

This project addresses these challenges through AI-powered automated proposal analysis and intelligent recommendation.

---

# 🚀 Features

### 📄 Proposal Management

- Upload Grant Proposal (PDF/DOCX/TXT)
- Automatic document validation
- OCR support for scanned documents

### 🧠 NLP Processing

- Tokenization
- Stop-word Removal
- Lemmatization
- Keyword Extraction
- Domain Identification
- TF-IDF Vectorization

### 🤖 Machine Learning

- Proposal Similarity Analysis
- Social Impact Prediction
- Recommendation Engine
- Domain Classification

### 📊 Analytics Dashboard

- Social Impact Score
- Recommendation Result
- Proposal Analysis
- Keyword Visualization
- Evaluation Reports

### 📑 Report Generation

- Download Evaluation Reports
- Proposal Summary
- Recommendation Report
- Explainable AI Insights

---

# ⚙️ System Workflow

```text
Grant Proposal Upload
          │
          ▼
Document Processing
          │
          ▼
Text Preprocessing
(Tokenization, Stopword Removal,
Lemmatization)
          │
          ▼
TF-IDF Feature Extraction
          │
          ▼
Cosine Similarity Analysis
          │
          ▼
Domain Identification
          │
          ▼
Social Impact Prediction
          │
          ▼
Grant Recommendation
          │
          ▼
Evaluation Dashboard
```

---

# 🧠 Machine Learning Pipeline

The recommendation engine follows a complete NLP workflow.

### Step 1

Proposal Upload

↓

### Step 2

Text Cleaning

- Remove punctuation
- Lowercase conversion
- Stop-word removal
- Lemmatization

↓

### Step 3

TF-IDF Vectorization

The cleaned proposal is converted into numerical vectors using the TF-IDF algorithm.

↓

### Step 4

Cosine Similarity

The proposal is compared against available grant descriptions and funding schemes.

↓

### Step 5

Recommendation Engine

The system ranks matching grants according to similarity score.

↓

### Step 6

Impact Prediction

Machine Learning models predict the expected social impact.

↓

### Step 7

Final Decision

- ✅ Approve
- 🟡 Partially Fund
- ❌ Reject

---

# 🏗 System Architecture

The framework consists of five major layers:

### 📥 Input Layer

- Proposal Upload
- Organization Details
- Document Acquisition

### 📝 NLP Processing Layer

- OCR
- Text Cleaning
- Keyword Extraction
- TF-IDF
- Feature Engineering

### 🤖 ML Prediction Layer

- Similarity Analysis
- Social Impact Prediction
- Classification

### 📈 Recommendation Engine

- Grant Matching
- Eligibility Evaluation
- Recommendation Generation

### 📊 Dashboard

- Reports
- Recommendation Results
- Analytics
- Explainability

---

# 📂 Modules

### 👤 User Module

- Register/Login
- Upload Proposal
- View Results
- Download Reports

### 🗂 Proposal Module

- Proposal Storage
- Proposal Management
- Document Validation

### 🧠 NLP Module

- Text Extraction
- Text Cleaning
- Keyword Extraction
- TF-IDF

### 🤖 Recommendation Module

- Similarity Matching
- Grant Ranking
- Eligibility Prediction

### 📊 Analytics Module

- Dashboard
- Reports
- Statistics

### 🔐 Admin Module

- Dataset Management
- Model Retraining
- User Management

---

# 🛠 Technologies Used

## Programming Language

- Python

## Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap

## Backend

- Flask

## Machine Learning

- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## NLP

- NLTK
- SpaCy

## Database

- MySQL

## Tools

- Git
- GitHub
- VS Code

---

# 📂 Project Structure

```text
Grant-Analysis-System/
│
├── dataset/
│
├── models/
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│
├── templates/
│
├── uploads/
│
├── reports/
│
├── app.py
├── requirements.txt
├── README.md
└── model.pkl
```

---

# 📊 Output

The system generates:

- Grant Recommendation
- Social Impact Score
- Proposal Similarity Score
- Proposal Domain
- Eligibility Status
- Evaluation Report
- Keyword Analysis

---

# 📈 Benefits

✅ Reduces manual evaluation time

✅ Improves consistency

✅ Reduces reviewer bias

✅ Supports transparent decision-making

✅ Enhances grant allocation efficiency

✅ Enables scalable proposal evaluation

---

# 🎯 Target Users

- Government Agencies
- NGOs
- CSR Organizations
- Non-Profit Foundations
- Academic Institutions
- Research Organizations

---

# 🔒 Non-Functional Features

- Secure Authentication
- Modular Architecture
- Fast Processing
- Explainable AI
- Scalable Design
- Responsive User Interface

---

# 🔮 Future Enhancements

- BERT-based Semantic Search
- Large Language Model Integration
- Multi-language Proposal Support
- Fraud Detection
- Real-time Government Grant APIs
- Cloud Deployment
- Advanced Analytics Dashboard
- AI Chat Assistant for Proposal Review

---

# 📸 Screenshots

```
screenshots/

├── home.png
├── login.png
├── upload.png
├── dashboard.png
├── recommendation.png
├── report.png
```

Add screenshots here.

---

# 📚 Research Concepts

- Natural Language Processing
- TF-IDF
- Cosine Similarity
- Machine Learning
- Information Retrieval
- Feature Engineering
- Explainable AI (XAI)

---

# 🤝 Contributing

Contributions are welcome.

```bash
Fork the repository

Create a feature branch

git checkout -b feature-name

Commit your changes

git commit -m "Added feature"

Push to GitHub

git push origin feature-name

Open a Pull Request
```

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Authors

**R. Shruthi Sagar**

Department of Computer Science & Engineering (Data Science)

Atria Institute of Technology

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

It helps others discover the project and supports future development.
