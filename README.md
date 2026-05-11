# Student Performance Predictor with Ollama AI

A machine learning and AI-powered Flask web application that predicts student academic performance using the Student Performance Dataset. The application combines a Random Forest machine learning model with Ollama AI-generated explanations to create an interactive educational analytics tool.

---

## Project Overview

This project was developed in Python using Flask and Scikit-learn. The application allows users to enter student academic information and receive:

* Predicted student performance category
* Machine learning confidence score
* Student academic risk score
* AI-generated academic explanation using Ollama

The goal of the project was to demonstrate the integration of:

* Machine learning
* Flask web development
* Local large language models (LLMs)
* Predictive analytics
* AI-assisted educational insights

---

## Technologies Used

* Python
* Flask
* Pandas
* Scikit-learn
* Random Forest Classifier
* Ollama
* HTML/CSS
* Cursor IDE

---

## Features

### Machine Learning Prediction

The application uses a Random Forest Classifier trained on the student performance dataset to predict student outcomes.

### Student Risk Score

A custom risk score is calculated based on:

* Study time
* Previous failures
* Absences
* Prior grades

### Ollama AI Integration

The app connects to a locally running Ollama LLM (`llama3.2`) to generate supportive AI explanations for each prediction.

### Interactive Flask Web App

Users can interact with the application through a modern web interface built with Flask and custom CSS styling.

---

## Dataset

Dataset Used:

* `student-mat.csv`

Dataset contains student academic and demographic variables including:

* Study time
* Absences
* Previous grades
* Failures
* Family support
* Internet access
* Final grades

Target variable:

* Student performance category derived from final grade (`G3`)

Performance Categories:

* At Risk
* Average
* High Performing

---

## Machine Learning Workflow

1. Load and preprocess student dataset
2. Create performance classification labels
3. Split training and testing data
4. Train Random Forest model
5. Generate predictions
6. Calculate confidence probabilities
7. Generate AI explanations through Ollama

---

## Running the Application

### Install Requirements

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama run llama3.2
```

### Run Flask Application

```bash
python3 app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
student-performance-ai/
│
├── app.py
├── requirements.txt
├── student-mat.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

## Educational Purpose

This project was created as part of a graduate-level Data Science and Artificial Intelligence course project focused on:

AI engineering concepts
predictive analytics
machine learning deployment
local LLM integration
Flask application development
