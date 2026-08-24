from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_recommend_valid_customer():
    response = client.post("/recommend", json={
        "customer_id": "CUST00001",
        "top_n": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST00001"
    assert len(data["recommendations"]) == 5


def test_recommend_invalid_customer():
    response = client.post("/recommend", json={
        "customer_id": "INVALID_CUSTOMER",
        "top_n": 5
    })
    assert response.status_code == 404


def test_recommend_invalid_top_n():
    response = client.post("/recommend", json={
        "customer_id": "CUST00001",
        "top_n": 0
    })
    assert response.status_code == 422


def test_sentiment_positive():
    response = client.post("/sentiment", json={
        "review_text": "Excellent product, I am very happy with it."
    })
    assert response.status_code == 200
    assert response.json()["sentiment"] in {"positive", "neutral", "negative"}


def test_sentiment_empty():
    response = client.post("/sentiment", json={
        "review_text": ""
    })
    assert response.status_code == 422


def test_review_analysis():
    response = client.post("/review-analysis", json={
        "product_name": "Zenith Laptop Product 225",
        "reviews": [
            {
                "sentiment": "positive",
                "review_text": "Excellent performance and very easy to use."
            },
            {
                "sentiment": "negative",
                "review_text": "The build quality is disappointing."
            }
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "common_complaints" in data
    assert "praised_features" in data
    assert "business_insights" in data


def test_review_analysis_empty_reviews():
    response = client.post("/review-analysis", json={
        "product_name": "Test Product",
        "reviews": []
    })
    assert response.status_code == 422