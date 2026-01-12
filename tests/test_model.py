from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint returns correct status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Active"
    assert "model" in data
    assert "metrics" in data

def test_predict_cluster():
    """Test the prediction endpoint returns a cluster and metadata."""
    payload = {"text": "Stock market hits record high as tech shares rally"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "cluster" in data
    assert isinstance(data["cluster"], int)
    assert "silhouette_score" in data
    assert "number_of_clusters" in data

def test_predict_invalid_input():
    """Test that missing required fields returns 422 error."""
    response = client.post("/predict", json={})
    assert response.status_code == 422