# AI-Powered Product Recommendation & Review Analysis System

An end-to-end AI/ML system for **personalized product recommendation, customer interaction analysis, sentiment classification, and intelligent review insights**.

The project combines traditional machine learning, recommendation techniques, experiment tracking, model management, LLM-based analysis, REST APIs, and containerized deployment into a complete production-oriented pipeline.

---

## 1. Project Overview

E-commerce platforms generate large volumes of customer interaction and review data. Analyzing this data can help businesses understand customer preferences, identify product sentiment, and provide more relevant product recommendations.

This project builds an AI-powered system that:

* Recommends products based on customer preferences and interactions.
* Analyzes customer reviews using machine learning.
* Classifies reviews by sentiment.
* Compares multiple machine learning models.
* Tracks experiments using **MLflow**.
* Registers the best-performing model using the **MLflow Model Registry**.
* Uses an LLM to generate higher-level insights from customer reviews.
* Provides REST APIs using **FastAPI**.
* Runs as a containerized application using **Docker**.
* Includes automated testing for core components.

---

## 2. Problem Statement

Modern e-commerce platforms collect information from multiple sources, including:

* Customer profiles
* Product catalogs
* Product interactions
* Purchases
* Ratings
* Reviews
* Browsing behavior

Simply storing this data does not provide personalized value to customers or actionable insights to businesses.

The objective of this project is to develop an integrated AI/ML solution that uses customer interaction history and review data to:

1. Recommend relevant products.
2. Understand customer sentiment.
3. Extract meaningful insights from reviews.
4. Track and manage ML experiments.
5. Provide the functionality through production-style APIs.

---

## 3. Objectives

### Primary Objectives

* Build a product recommendation engine.
* Build a customer review sentiment classification system.
* Compare multiple ML algorithms.
* Evaluate models using appropriate performance metrics.
* Track experiments using MLflow.
* Register the selected production model.
* Generate LLM-based summaries and insights from reviews.
* Develop FastAPI endpoints for application access.
* Containerize the application using Docker.
* Build automated tests for important system components.

### Secondary Objectives

* Maintain a modular and maintainable project structure.
* Separate data processing, ML training, inference, and API logic.
* Make the system reproducible.
* Provide clear documentation.
* Follow production-oriented ML engineering practices where practical.

---

# 4. System Architecture

The system follows a modular pipeline:

```text
                    ┌──────────────────────┐
                    │      Raw Dataset     │
                    │ Customers / Products │
                    │ Interactions/Reviews │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Processing    │
                    │ Cleaning & Validation │
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐       ┌──────────────────┐
       │ Recommendation   │       │ Sentiment        │
       │ System           │       │ Classification   │
       └────────┬─────────┘       └────────┬─────────┘
                │                          │
                │                          ▼
                │                 ┌──────────────────┐
                │                 │ ML Model Training│
                │                 │ & Evaluation     │
                │                 └────────┬─────────┘
                │                          │
                │                          ▼
                │                 ┌──────────────────┐
                │                 │      MLflow      │
                │                 │ Tracking/Registry│
                │                 └────────┬─────────┘
                │                          │
                └─────────────┬────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │     FastAPI Layer    │
                    │ REST API Endpoints   │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌──────────────────┐         ┌──────────────────┐
       │ Recommendation   │         │ Review Analysis  │
       │ API              │         │ API              │
       └──────────────────┘         └────────┬─────────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │       LLM        │
                                   │ Review Insights  │
                                   └──────────────────┘

                         Docker Container
```

---

# 5. Project Structure

The project is organized into separate modules for data generation, machine learning, API development, testing, and deployment.

```text
AI_Product_Recommendation/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── data_generation/
│   └── generate_dataset.py
│
├── models/
│   ├── recommendation/
│   └── sentiment/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data_processing/
│   ├── recommendation/
│   ├── sentiment/
│   ├── llm/
│   └── utils/
│
├── tests/
│   ├── test_api.py
│   ├── test_recommendation.py
│   └── test_sentiment.py
│
├── mlruns/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> The final structure may evolve during implementation if a structural change provides a significant improvement in maintainability, reproducibility, or functionality.

---

# 6. Dataset

The system uses a structured e-commerce dataset containing information about customers, products, interactions, and reviews.

## Dataset Components

### Customers

Contains customer-level information used to identify customer preferences.

Example fields:

```text
customer_id
age
gender
location
```

### Products

Contains product catalog information.

Example fields:

```text
product_id
product_name
category
brand
price
description
```

### Interactions

Contains customer-product interaction history.

Example interactions include:

```text
view
click
cart
wishlist
purchase
rating
```

Example fields:

```text
customer_id
product_id
interaction_type
timestamp
```

### Reviews

Contains customer reviews and associated ratings.

Example fields:

```text
review_id
customer_id
product_id
rating
review_text
timestamp
```

## Dataset Generation

A synthetic dataset can be generated using:

```bash
python data_generation/generate_dataset.py
```

The generated data is stored under:

```text
data/raw/
```

Processed datasets are stored under:

```text
data/processed/
```

---

# 7. Data Processing

Before model training, the raw data goes through a preprocessing pipeline.

### Main processing steps

1. Load raw datasets.
2. Validate required columns.
3. Handle missing values.
4. Remove duplicate records.
5. Normalize categorical values.
6. Validate customer and product IDs.
7. Process timestamps.
8. Encode categorical variables where required.
9. Generate recommendation features.
10. Prepare text data for sentiment analysis.
11. Split data into training and evaluation datasets.

The preprocessing pipeline is designed to be reproducible and reusable.

---

# 8. Recommendation System

The recommendation engine provides personalized product recommendations based on customer behavior and product information.

## Recommendation Approach

The system can combine multiple recommendation strategies:

### Collaborative Filtering

Uses customer-product interaction history to identify products that similar customers interacted with.

```text
Customer → Interaction History → Similar Customers → Recommended Products
```

### Content-Based Recommendation

Uses product attributes such as:

* Category
* Brand
* Description
* Price
* Product features

to identify products similar to those previously preferred by a customer.

### Hybrid Recommendation

The final system can combine collaborative and content-based signals to improve recommendation quality.

```text
Customer History
       │
       ├───────────────┐
       ▼               ▼
Collaborative      Content-Based
Filtering          Recommendation
       │               │
       └───────┬───────┘
               ▼
       Recommendation
          Ranking
               │
               ▼
       Top-N Products
```

## Recommendation Output

Example:

```json
{
  "customer_id": "CUST00001",
  "recommendations": [
    {
      "product_id": "PROD00125",
      "score": 0.92
    },
    {
      "product_id": "PROD00341",
      "score": 0.87
    }
  ]
}
```

---

# 9. Sentiment Analysis

The sentiment analysis component classifies customer reviews into sentiment categories.

Possible classes:

```text
Positive
Neutral
Negative
```

## Machine Learning Pipeline

```text
Review Text
     │
     ▼
Text Cleaning
     │
     ▼
Feature Extraction
     │
     ▼
ML Model
     │
     ▼
Sentiment Prediction
```

Possible feature extraction approaches include:

* TF-IDF
* N-grams
* Other suitable text representations

## Models

Multiple classification models can be evaluated, such as:

* Logistic Regression
* Naive Bayes
* Linear SVM
* Other suitable baseline models

The final model will be selected based on actual validation results rather than assumptions.

---

# 10. Model Evaluation

The sentiment models will be evaluated using appropriate classification metrics.

Primary metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

For imbalanced datasets, additional emphasis will be placed on:

* Macro F1-score
* Class-level precision and recall

Example evaluation:

```text
Model                  Accuracy    F1 Score
------------------------------------------------
Logistic Regression       --         --
Naive Bayes               --         --
Linear SVM                --         --
```

Actual values will be populated after model training.

---

# 11. MLflow Experiment Tracking

MLflow is used to track machine learning experiments and manage model versions.

The system records information such as:

* Model name
* Hyperparameters
* Training metrics
* Validation metrics
* Dataset/version information
* Model artifacts

Example workflow:

```text
Model Training
      │
      ▼
MLflow Experiment
      │
      ├── Parameters
      ├── Metrics
      ├── Artifacts
      └── Model
             │
             ▼
       Model Registry
             │
             ▼
     Selected Model Version
```

## MLflow Components

### Experiment Tracking

Each training run records parameters and metrics.

### Model Registry

The selected model is registered and versioned.

### Model Lifecycle

A typical lifecycle is:

```text
Training
   ↓
Evaluation
   ↓
MLflow Run
   ↓
Model Registration
   ↓
Validation
   ↓
Production Model
```

---

# 12. LLM-Based Review Analysis

In addition to traditional sentiment classification, an LLM is used to generate higher-level insights from customer reviews.

The LLM component can provide:

* Review summaries
* Main customer complaints
* Frequently mentioned positive aspects
* Product strengths
* Product weaknesses
* Feature requests
* Common customer concerns
* Overall review insights

Example:

```text
Input:
Multiple customer reviews for a product

        ↓

LLM Analysis

        ↓

Output:
• Customers like the product quality.
• Battery life is frequently praised.
• Several customers report charging issues.
• Customers are requesting improved durability.
```

The LLM is intended to complement the ML sentiment classifier rather than replace it.

---

# 13. FastAPI

FastAPI provides the REST API layer for accessing the system.

## Planned Endpoints

### Health Check

```http
GET /health
```

Returns the API status.

### Product Recommendations

```http
GET /recommend/{customer_id}
```

Returns personalized recommendations.

### Sentiment Prediction

```http
POST /predict-sentiment
```

Accepts review text and returns sentiment.

Example request:

```json
{
  "review": "The product quality is excellent and I really enjoyed using it."
}
```

Example response:

```json
{
  "sentiment": "positive"
}
```

### Review Analysis

```http
POST /analyze-review
```

Uses the LLM component to generate higher-level review insights.

### API Documentation

FastAPI automatically provides interactive documentation through:

```text
/docs
```

and:

```text
/redoc
```

when the application is running.

---

# 14. Docker

The application is containerized using Docker to provide a consistent runtime environment.

The Docker container packages:

* Python runtime
* Application code
* Dependencies
* API
* ML inference components

Example workflow:

```text
Source Code
     │
     ▼
Docker Build
     │
     ▼
Docker Image
     │
     ▼
Docker Container
     │
     ▼
FastAPI Application
```

Build the image:

```bash
docker build -t ai-product-recommendation .
```

Run the container:

```bash
docker run -p 8000:8000 ai-product-recommendation
```

The API can then be accessed through:

```text
http://localhost:8000
```

---

# 15. Testing

The project uses automated testing to validate important components.

Testing areas include:

* Data processing
* Recommendation logic
* Sentiment prediction
* API endpoints
* Input validation
* Model loading
* Health checks

Run the test suite with:

```bash
pytest -v
```

Example:

```bash
pytest tests/ -v
```

The goal is to ensure that changes to one component do not unintentionally break another component.

---

# 16. Environment Configuration

Sensitive configuration values should not be hard-coded into the source code.

Create a local environment file based on:

```text
.env.example
```

Example:

```env
GEMINI_API_KEY=your_api_key
MLFLOW_TRACKING_URI=http://localhost:5000
```

The actual `.env` file should **not** be committed to Git.

Make sure `.gitignore` contains:

```text
.env
.venv/
__pycache__/
mlruns/
*.pyc
```

---

# 17. Installation

## Prerequisites

The following software is required:

* Python 3.11+
* pip
* Git
* Docker
* MLflow

Clone the repository:

```bash
git clone <repository-url>
cd AI_Product_Recommendation
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 18. Running the Project

## Step 1 — Generate Dataset

```bash
python data_generation/generate_dataset.py
```

## Step 2 — Process Data

Run the appropriate preprocessing pipeline.

```bash
python <data-processing-script>
```

## Step 3 — Train Models

Train the recommendation and sentiment models.

```bash
python <training-script>
```

## Step 4 — Start MLflow

```bash
mlflow ui
```

MLflow will normally be available at:

```text
http://localhost:5000
```

## Step 5 — Start FastAPI

```bash
uvicorn api.main:app --reload
```

API:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# 19. End-to-End Workflow

The complete system follows this workflow:

```text
                ┌───────────────┐
                │ Raw E-Commerce│
                │     Data      │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Data Cleaning │
                │ & Processing  │
                └───────┬───────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
   ┌────────────────┐      ┌────────────────┐
   │ Recommendation │      │ Review         │
   │ Pipeline       │      │ Pipeline       │
   └───────┬────────┘      └───────┬────────┘
           │                       │
           │                       ▼
           │               ┌────────────────┐
           │               │ Sentiment ML   │
           │               │ Classification │
           │               └───────┬────────┘
           │                       │
           │                       ▼
           │               ┌────────────────┐
           │               │ MLflow         │
           │               │ Tracking       │
           │               └───────┬────────┘
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ FastAPI REST API │
              └────────┬─────────┘
                       │
             ┌─────────┴──────────┐
             │                    │
             ▼                    ▼
      Recommendations      Review Analysis
                                  │
                                  ▼
                           ┌──────────────┐
                           │     LLM      │
                           │   Insights   │
                           └──────────────┘
```

---

# 20. Results

Results will be added after the complete implementation and evaluation.

The final report will include:

### Recommendation Results

* Precision@K
* Recall@K
* Hit Rate
* Recommendation examples

### Sentiment Results

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### MLflow Results

* Experiment comparison
* Best-performing model
* Registered model version

### LLM Results

* Review summaries
* Key customer concerns
* Product strengths
* Product weaknesses

Actual metrics will be reported from experiments and will not be manually estimated.

---

# 21. Limitations

Potential limitations include:

* Synthetic data may not fully represent real-world customer behavior.
* Recommendation quality depends on the amount and quality of interaction data.
* New customers may suffer from the cold-start problem.
* New products may suffer from limited interaction history.
* Sentiment classification may struggle with sarcasm or ambiguous language.
* LLM-generated insights may occasionally contain inaccurate interpretations.
* LLM usage may introduce API costs and latency.
* MLflow and API deployment may require additional infrastructure for production-scale usage.

---

# 22. Future Improvements

Potential future improvements include:

* Deep-learning-based recommendation models.
* Neural collaborative filtering.
* Transformer-based sentiment classification.
* Retrieval-Augmented Generation (RAG) for review analysis.
* Real-time recommendation updates.
* User-specific recommendation explanations.
* Advanced ranking models.
* Online model monitoring.
* Data drift detection.
* Model performance monitoring.
* A production database instead of file-based datasets.
* Cloud deployment.
* Authentication and authorization for APIs.
* CI/CD automation.
* Kubernetes-based deployment for large-scale production environments.

---

# 23. Technology Stack

| Component            | Technology                             |
| -------------------- | -------------------------------------- |
| Programming Language | Python                                 |
| Data Processing      | Pandas, NumPy                          |
| Machine Learning     | Scikit-learn                           |
| Recommendation       | Collaborative / Content-Based / Hybrid |
| Experiment Tracking  | MLflow                                 |
| Model Registry       | MLflow Model Registry                  |
| LLM                  | Gemini / Compatible LLM                |
| API                  | FastAPI                                |
| API Server           | Uvicorn                                |
| Testing              | Pytest                                 |
| Containerization     | Docker                                 |
| Version Control      | Git                                    |
| Repository           | GitHub                                 |

The final technology stack may be adjusted if experimentation demonstrates that another approach provides a substantial improvement.

---

# 24. Project Status

| Component                | Status         |
| ------------------------ | -------------- |
| Project Architecture     | 🟡 In Progress |
| Dataset Generation       | 🟡 In Progress |
| Data Processing          | 🟡 In Progress |
| Recommendation System    | ⬜ Planned      |
| Sentiment Classification | ⬜ Planned      |
| Model Comparison         | ⬜ Planned      |
| MLflow Tracking          | ⬜ Planned      |
| MLflow Model Registry    | ⬜ Planned      |
| LLM Review Analysis      | ⬜ Planned      |
| FastAPI                  | ⬜ Planned      |
| Docker                   | ⬜ Planned      |
| Testing                  | 🟡 In Progress |
| Documentation            | 🟡 In Progress |
| Final Results            | ⬜ Pending      |

Status will be updated as each component is completed.

---

# 25. Key Features

### 🤖 AI/ML

* Personalized product recommendations
* Sentiment classification
* Multiple ML model comparison
* LLM-powered review insights

### 📊 MLOps

* MLflow experiment tracking
* Model versioning
* Model Registry
* Reproducible experiments

### 🚀 Backend

* FastAPI REST APIs
* Automatic API documentation
* Modular application architecture

### 🐳 Deployment

* Docker containerization
* Reproducible runtime environment

### 🧪 Quality

* Automated testing
* Input validation
* Modular components

---

# 26. Security Considerations

The project follows basic security practices:

* API keys are stored using environment variables.
* `.env` files are excluded from version control.
* Secrets are not hard-coded.
* API inputs are validated.
* Dependencies are maintained through `requirements.txt`.

For production deployment, additional measures such as authentication, authorization, HTTPS, rate limiting, secret management, and API monitoring should be implemented.

---

# 27. Conclusion

The **AI-Powered Product Recommendation & Review Analysis System** demonstrates how multiple AI/ML and software engineering components can be integrated into a single end-to-end application.

The project combines:

```text
Data
 ↓
Machine Learning
 ↓
Recommendation
 ↓
Sentiment Analysis
 ↓
MLflow
 ↓
LLM Analysis
 ↓
FastAPI
 ↓
Docker
```

The final system is intended to demonstrate not only individual machine learning models, but also the complete workflow required to transform ML models into an accessible, testable, and deployable application.

---

## Author

**Shubhranshu Verma**

AI/ML | Software Engineering | Python | Machine Learning

---

## License

This project is developed for educational and portfolio purposes.
