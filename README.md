# Symptom Recommendation System

A Python-based symptom recommendation system using TF-IDF and K-Nearest Neighbors (KNN) to suggest related symptoms for patients based on their search terms. This project include a **FastAPI** endpoint for easy integration.

---

## Features

- Preprocesses symptom and patient data from CSV files
- Converts symptoms and search terms into numerical vectors using **TF-IDF**
- Considers patient demographic features (gender and age)
- Provides recommendations based on:
  1. Direct symptom-to-symptom mapping
  2. Similar patients using KNN
- Combines scores from mapping and similar patients to return the top recommendations
- API-ready with FastAPI for integration

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/symptom-recommendation.git
cd symptom-recommendation
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Data
- The Dataset is **Confidential**

## How to Run
```bash
python main.py
```

## API
- There are two endpoint right now, which are...
- Health check: GET /health -- for check the fastapi if they are running normally.
- Predict symptoms: POST /predict -- for input the patient gender, age and symptoms.

## You can access locally by enter http://127.0.0.1:8000/ with "health" or "predict" path.

## Example Input
```bash
{
  "gender": "Male",
  "age": 26,
  "symptoms": "ไข้"
}
```


# Example Results
```bash
{
  "success": true,
  "input": {
    "gender": "Male",
    "age": 26,
    "symptoms": "ไข้"
  },
  "similiar symptoms": ["ไอ", "เจ็บคอ", "ปวดหัว"]
}
```
<img width="343" height="609" alt="image" src="https://github.com/user-attachments/assets/a0888a80-7fbb-43cd-a0fb-34d26c2bc9cb" />


# Thank for visiting my respository XD




