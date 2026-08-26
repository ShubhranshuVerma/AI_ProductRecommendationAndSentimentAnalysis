# AI-Powered Product Recommendation & Review Analysis System

An end-to-end AI/ML capstone project that combines:

- Personalized product recommendations
- Machine-learning-based sentiment classification
- Transformer-based sentiment analysis using CardiffNLP RoBERTa
- LLM-powered customer review analysis
- MLflow experiment tracking and model registration
- FastAPI REST APIs
- Streamlit user interface
- Docker deployment
- Automated testing

The system is designed as a compact, explainable, and production-oriented demonstration of how multiple AI components can work together in an e-commerce application.

---

# 1. Project Overview

E-commerce platforms generate large amounts of customer interaction and review data.

This project addresses two related problems:

1. **What products should be recommended to a customer?**
2. **What can we learn from customer reviews?**

The system combines recommendation algorithms, sentiment classification, and LLM-based review analysis into a single application.

The final system provides:

- Personalized product recommendations
- Positive, neutral, or negative sentiment classification
- Sentiment confidence scores
- AI-generated review summaries
- Common customer complaints
- Praised product features
- Business insights derived from reviews

---

# 2. Problem Statement

Modern e-commerce platforms need to understand both:

- Customer behavior
- Customer feedback

Traditional recommendation systems can identify products a customer may be interested in, but they do not explain how customers feel about products.

Similarly, sentiment analysis can classify reviews but does not directly provide personalized recommendations.

This project integrates both capabilities into a single system.

### Main objectives

- Build a recommendation system using customer-product interactions.
- Compare the recommendation system against a popularity baseline.
- Implement sentiment classification using a transformer model.
- Evaluate sentiment predictions on labeled data.
- Register the production sentiment model with MLflow.
- Analyze customer reviews using an LLM.
- Expose the functionality through REST APIs.
- Provide a simple Streamlit interface.
- Containerize the backend using Docker.
- Validate the application with automated tests.

---

# 3. System Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit UI    │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
                                    │ HTTP
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI API     │
                         │      api/main.py     │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ Recommendation │ │   Sentiment    │ │ Review Analysis│
        │    Service     │ │    Service     │ │      LLM       │
        └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                │                  │                  │
                ▼                  ▼                  ▼
        Customer/Product     CardiffNLP RoBERTa   Google GenAI
          Interactions        Transformer Model       API
                │                  │
                │                  │
                ▼                  ▼
        Recommendation       Sentiment Result
            Results          + Confidence
                         ┌──────────────────────┐
                         │       MLflow         │
                         │ Tracking + Registry  │
                         └──────────────────────┘
````

---

# 4. Main Components

## 4.1 Recommendation System

The recommendation component uses customer interaction history and product information to generate personalized recommendations.

The system also includes a popularity-based baseline for comparison.

### Recommendation output

Each recommendation contains:

* Product ID
* Product name
* Product category
* Recommendation score

Example:

```json
{
  "customer_id": "CUST00001",
  "recommendations": [
    {
      "product_id": "PROD00225",
      "product_name": "Zenith Laptop Product 225",
      "category": "Electronics",
      "recommendation_score": 0.5519
    }
  ]
}
```

### Recommendation evaluation

The recommendation system is evaluated using:

* Precision@10
* Recall@10
* Hit Rate@10
* Coverage

The current evaluation produced:

| Model               | Precision@10 | Recall@10 | Hit Rate@10 | Coverage |
| ------------------- | -----------: | --------: | ----------: | -------: |
| Hybrid Recommender  |       0.0191 |    0.1042 |      0.1760 |    0.580 |
| Popularity Baseline |       0.0087 |    0.0472 |      0.0859 |    0.068 |

The hybrid recommender outperformed the popularity baseline on all reported evaluation metrics.

---

# 5. Sentiment Analysis

The production sentiment classifier uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The model predicts three sentiment classes:

```text
negative
neutral
positive
```

The transformer model is loaded through:

```text
sentiment/transformer_service.py
```

The FastAPI sentiment endpoint uses this service directly.

### Example

Input:

```text
I am extremely satisfied with this purchase.
```

Output:

```json
{
  "sentiment": "positive",
  "confidence": 0.9884,
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

Another example:

```text
The device overheats constantly and I regret buying it.
```

Output:

```text
negative
```

And:

```text
The product is okay, nothing special.
```

Output:

```text
neutral
```

---

# 6. Sentiment Model Evaluation

The sentiment dataset is stored under:

```text
data/processed/sentiment/
```

with:

```text
train.csv
test.csv
```

The model evaluation includes:

* Accuracy
* Macro F1
* Weighted F1
* Precision
* Recall
* Per-class metrics
* Classification report

The production transformer model achieved the following evaluation results on 3,000 test samples:

```text
Accuracy       : 0.7427
Macro F1       : 0.7325
Weighted F1    : 0.7325
Negative Recall: 0.8160
```

Per-class results:

| Sentiment | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Negative  |      0.75 |   0.82 | 0.78 |
| Neutral   |      0.73 |   0.51 | 0.60 |
| Positive  |      0.74 |   0.90 | 0.81 |

The model provides a reasonable three-class sentiment classification baseline while remaining compact enough for the capstone application.

---

# 7. MLflow

MLflow is used to support the machine-learning lifecycle.

The project uses MLflow for:

* Experiment tracking
* Model logging
* Model registration
* Model version management

The registered model is:

```text
ProductSentimentModel
```

The project uses a local SQLite MLflow tracking database:

```text
mlflow.db
```

The MLflow tracking utilities are located in:

```text
mlflow_tracking/
```

The training workflow is implemented in:

```text
sentiment/train_models.py
```

Model registration is handled by:

```text
sentiment/register_model.py
```

The production API does not depend on the old serialized `joblib` sentiment model. Sentiment inference is handled directly by the transformer service.

---

# 8. LLM-Based Review Analysis

The review-analysis component uses a generative AI model to transform customer reviews into useful business insights.

The LLM component is located in:

```text
llm/
├── client.py
└── review_analysis.py
```

The system analyzes:

* Review sentiment
* Review text
* Product information

and generates:

### Summary

A concise summary of overall customer feedback.

### Common complaints

Issues frequently mentioned by customers.

### Praised features

Features customers respond positively to.

### Business insights

Actionable recommendations for the business.

Example output:

```json
{
  "summary": "Customers appreciate the laptop's performance and battery life, but express concern over thermal management during prolonged use.",
  "common_complaints": [
    "Laptop gets hot during extended use"
  ],
  "praised_features": [
    "Excellent performance",
    "Battery life"
  ],
  "business_insights": [
    "Highlight performance and battery efficiency as key selling points in marketing.",
    "Address thermal management issues to improve customer satisfaction during long usage sessions."
  ]
}
```

---

# 9. FastAPI

The backend exposes the main functionality through REST APIs.

The API application is located at:

```text
api/main.py
```

Routes are organized under:

```text
api/routes/
```

Available endpoints include:

```text
GET  /health
POST /recommend
POST /sentiment
POST /review-analysis
```

---

# 10. API Examples

## Health Check

```bash
curl -s http://127.0.0.1:8001/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

## Sentiment Prediction

```bash
curl -s -X POST \
  http://127.0.0.1:8001/sentiment \
  -H 'Content-Type: application/json' \
  -d '{
    "review_text": "Excellent product, I am very happy with it."
  }'
```

Example response:

```json
{
  "sentiment": "positive",
  "confidence": 0.9884,
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

---

## Product Recommendation

```bash
curl -s -X POST \
  http://127.0.0.1:8001/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "CUST00001",
    "top_n": 5
  }'
```

---

## Review Analysis

```bash
curl -s -X POST \
  http://127.0.0.1:8001/review-analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "product_name": "Sample Laptop",
    "reviews": [
      {
        "sentiment": "positive",
        "review_text": "Excellent performance and battery life."
      },
      {
        "sentiment": "negative",
        "review_text": "The laptop gets hot during long usage."
      }
    ]
  }'
```

---

# 11. Streamlit Interface

The project includes a Streamlit frontend:

```text
app.py
```

The interface provides three main sections:

### Recommendations

Select a customer and retrieve personalized product recommendations.

### Sentiment Checker

Enter a review and receive:

* Sentiment
* Confidence
* Model information

### Review Analysis

Submit product reviews and receive:

* Summary
* Common complaints
* Praised features
* Business insights

The Streamlit application communicates with the FastAPI backend.

---

# 12. Dataset

The project uses synthetic e-commerce data generated for the capstone.

## Raw data

Located under:

```text
data/raw/
```

Files:

```text
customers.csv
interactions.csv
products.csv
reviews.csv
```

## Processed recommendation data

Located under:

```text
data/processed/recommendation/
```

Files:

```text
customers.csv
interactions.csv
products.csv
```

## Processed sentiment data

Located under:

```text
data/processed/sentiment/
```

Files:

```text
train.csv
test.csv
```

---

# 13. Data Generation

Synthetic e-commerce data can be generated using:

```text
data_generation/generate_dataset.py
```

Dataset validation is provided by:

```text
data_generation/validate_dataset.py
```

The generated data supports:

* Customer profiles
* Product catalog
* Customer-product interactions
* Customer reviews
* Sentiment labels

---

# 14. Project Structure

The current project structure is:

```text
AI_Product_Recommendation/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── __init__.py
│       ├── recommendation.py
│       ├── review_analysis.py
│       └── sentiment.py
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── interactions.csv
│   │   ├── products.csv
│   │   └── reviews.csv
│   │
│   └── processed/
│       ├── recommendation/
│       │   ├── customers.csv
│       │   ├── interactions.csv
│       │   └── products.csv
│       │
│       └── sentiment/
│           ├── train.csv
│           └── test.csv
│
├── data_generation/
│   ├── generate_dataset.py
│   └── validate_dataset.py
│
├── llm/
│   ├── __init__.py
│   ├── client.py
│   └── review_analysis.py
│
├── mlflow_tracking/
│   ├── __init__.py
│   └── tracking.py
│
├── recommendation/
│   ├── __init__.py
│   ├── evaluate_recommender.py
│   ├── recommender.py
│   └── service.py
│
├── sentiment/
│   ├── __init__.py
│   ├── natural_eval.csv
│   ├── natural_evaluation.py
│   ├── register_model.py
│   ├── train_models.py
│   └── transformer_service.py
│
├── tests/
│   ├── test_api.py
│   └── test_setup.py
│
├── app.py
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── mlflow.db
├── pytest.ini
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

Generated runtime directories such as:

```text
.venv/
__pycache__/
.pytest_cache/
mlruns/
artifacts/
```

are not part of the core source structure.

---

# 15. Installation

## Requirements

Recommended environment:

```text
Python 3.11
```

Create a virtual environment:

```bash
python3.11 -m venv .venv
```

Activate it:

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 16. Environment Variables

Create a `.env` file based on:

```text
.env.example
```

The project uses environment variables for configuration such as the LLM API key and application settings.

Example:

```text
LLM_API_KEY=your_api_key
```

Do not commit `.env` to Git.

---

# 17. Run the FastAPI Application Locally

Start the backend using:

```bash
uvicorn api.main:app --reload --port 8001
```

The API will be available at:

```text
http://127.0.0.1:8001
```

FastAPI documentation:

```text
http://127.0.0.1:8001/docs
```

---

# 18. Run the Streamlit Application

With the FastAPI backend running:

```bash
streamlit run app.py
```

The Streamlit application communicates with:

```text
http://127.0.0.1:8001
```

The API URL can be configured using:

```text
API_BASE_URL
```

---

# 19. Docker Deployment

The backend can be containerized using Docker.

Build the image:

```bash
docker build -t ai-product-recommendation .
```

Run the container:

```bash
docker rm -f ai-product-recommendation 2>/dev/null || true

docker run -d \
  -p 8001:8000 \
  --env-file .env \
  --name ai-product-recommendation \
  ai-product-recommendation
```

Check the container:

```bash
docker ps
```

Check logs:

```bash
docker logs --tail 100 ai-product-recommendation
```

Test the API:

```bash
curl -s http://127.0.0.1:8001/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

# 20. Docker Architecture

The Docker image contains:

* FastAPI application
* Recommendation system
* Transformer sentiment service
* LLM integration
* MLflow tracking utilities
* Processed application data
* MLflow database

The transformer model is downloaded through Hugging Face when required.

The Docker configuration uses CPU-based PyTorch for compatibility with lightweight Linux containers.

---

# 21. Testing

The project uses pytest.

Run the complete test suite:

```bash
pytest -v
```

Current validation:

```text
11 passed
```

Tests cover:

* Root endpoint
* Health endpoint
* Valid recommendation request
* Invalid customer handling
* Invalid recommendation parameters
* Sentiment prediction
* Empty sentiment input
* Review analysis
* Empty review analysis
* Python version
* Python environment

---

# 22. Recommendation Evaluation

The recommendation evaluation script can be executed with:

```bash
python recommendation/evaluate_recommender.py
```

It performs a customer-level train/test split and compares:

```text
Hybrid Recommender
vs.
Popularity Baseline
```

Metrics:

```text
Precision@10
Recall@10
Hit Rate@10
Coverage
```

Results are saved to:

```text
artifacts/recommendation/recommendation_comparison.csv
```

The `artifacts/` directory is generated during evaluation and does not need to be committed as source code.

---

# 23. Sentiment Training and Model Registration

The sentiment training workflow is implemented in:

```text
sentiment/train_models.py
```

The model registration workflow is implemented in:

```text
sentiment/register_model.py
```

The production sentiment model is:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The registered MLflow model name is:

```text
ProductSentimentModel
```

Model registration should be performed when creating or updating a model version, rather than during normal API inference.

---

# 24. Natural Evaluation

The project also includes:

```text
sentiment/natural_evaluation.py
```

and:

```text
sentiment/natural_eval.csv
```

These support evaluation against naturally written review examples.

They are maintained as evaluation/supporting artifacts and are separate from the production API inference path.

---

# 25. Model Selection

The sentiment component follows a practical model-development approach.

Traditional classification approaches such as:

* Logistic Regression
* Linear SVM
* Other classification baselines

can be evaluated during model development.

The final production sentiment model is:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

This provides contextual language understanding that is better suited to short, informal customer-review-style text.

The project therefore separates:

```text
Model experimentation
        ↓
Model evaluation
        ↓
Final model selection
        ↓
Production inference
```

---

# 26. Production Sentiment Inference

The production API does not use the previously serialized:

```text
models/sentiment_model.joblib
```

That legacy artifact has been removed.

Current inference is handled through:

```text
sentiment/transformer_service.py
```

This provides a single clear production path:

```text
FastAPI
   ↓
sentiment route
   ↓
transformer_service
   ↓
CardiffNLP RoBERTa
   ↓
sentiment + confidence
```

---

# 27. Error Handling

The API validates incoming requests using Pydantic schemas.

Examples of handled cases include:

* Empty review text
* Invalid customer IDs
* Invalid recommendation counts
* Empty review collections

The API returns appropriate HTTP responses instead of allowing invalid requests to silently fail.

---

# 28. Reproducibility

The project uses fixed random seeds where appropriate.

For example:

```text
RANDOM_STATE = 42
```

This helps keep dataset splitting and evaluation behavior reproducible.

---

# 29. Project Validation

The final application has been validated through multiple layers.

## API validation

```text
GET  /health              → PASS
POST /sentiment           → PASS
POST /recommend           → PASS
POST /review-analysis     → PASS
```

## Automated tests

```text
11 / 11 tests passed
```

## Docker

```text
Docker image build        → PASS
Container startup         → PASS
Health endpoint           → PASS
Sentiment endpoint        → PASS
Recommendation endpoint   → PASS
Review analysis endpoint  → PASS
```

## Recommendation evaluation

```text
Hybrid recommender        → PASS
Popularity baseline       → PASS
Metric comparison         → PASS
```

---

# 30. Limitations

The project is designed as a capstone demonstration rather than a large-scale production e-commerce platform.

Important limitations include:

* The dataset is synthetic.
* Recommendation quality depends on the generated interaction patterns.
* Transformer inference can be computationally expensive on CPU.
* The Hugging Face model may require an internet connection when not cached.
* LLM review analysis requires an external API and valid API credentials.
* MLflow is configured as a local tracking setup rather than a production tracking server.
* The recommendation model is intentionally compact and explainable rather than highly complex.
* The system does not implement distributed serving or large-scale feature stores.

---

# 31. Future Improvements

Possible future improvements include:

* Larger real-world datasets
* More advanced collaborative filtering
* Neural recommendation models
* Better cold-start strategies
* Model monitoring
* Automated model retraining
* Feature stores
* Production MLflow server
* Authentication and authorization
* API rate limiting
* Distributed deployment
* GPU-based transformer inference
* More comprehensive sentiment evaluation
* Multilingual sentiment analysis
* More advanced review summarization

These improvements are outside the scope of the current compact capstone implementation.

---

# 32. Key Technologies

| Technology         | Purpose                                |
| ------------------ | -------------------------------------- |
| Python 3.11        | Main programming language              |
| Pandas             | Data processing                        |
| NumPy              | Numerical operations                   |
| Scikit-learn       | ML algorithms and evaluation           |
| Transformers       | Transformer-based sentiment analysis   |
| PyTorch            | Transformer model execution            |
| CardiffNLP RoBERTa | Final sentiment model                  |
| FastAPI            | REST API                               |
| Pydantic           | API validation                         |
| Streamlit          | Frontend UI                            |
| MLflow             | Experiment tracking and model registry |
| Google GenAI       | LLM review analysis                    |
| Docker             | Containerization                       |
| Pytest             | Automated testing                      |

---

# 33. Quick Start

For the quickest way to run the project:

### Step 1 — Activate environment

```bash
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment

Create `.env`:

```text
LLM_API_KEY=your_api_key
```

### Step 4 — Run backend

```bash
uvicorn api.main:app --reload --port 8001
```

### Step 5 — Run frontend

In another terminal:

```bash
streamlit run app.py
```

### Step 6 — Or run with Docker

```bash
docker build -t ai-product-recommendation .

docker run -d \
  -p 8001:8000 \
  --env-file .env \
  --name ai-product-recommendation \
  ai-product-recommendation
```

### Step 7 — Verify

```bash
curl -s http://127.0.0.1:8001/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

# 34. Conclusion

This project demonstrates a complete AI-powered e-commerce workflow combining:

```text
Customer Data
     │
     ▼
Recommendation Engine
     │
     ▼
Personalized Products
```

and:

```text
Customer Reviews
     │
     ▼
Transformer Sentiment Analysis
     │
     ▼
Positive / Neutral / Negative
     │
     ▼
LLM Review Analysis
     │
     ▼
Business Insights
```

The components are exposed through a FastAPI backend, presented through a Streamlit interface, tracked using MLflow, and packaged for deployment using Docker.

The resulting system provides a compact and explainable demonstration of an end-to-end AI/ML product recommendation and customer feedback analysis platform.

````

### One important correction before you paste it

Your current Docker image **does not copy `models/` anymore**, which is correct because the old `models/sentiment_model.joblib` was deleted. The README above intentionally reflects that.

Also, don't manually create any of the old directories just to match the README. The README is now based on the **actual cleaned project structure** we validated.

After replacing it, run:

```bash
git diff --check
````

then:

```bash
pytest -v
```

and finally:

```bash
git status --short
```

At that point we'll have the codebase, documentation, tests, and Docker flow aligned.
