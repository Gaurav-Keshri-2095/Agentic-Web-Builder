from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient

def test_generate_missing_prompt(client: TestClient):
    response = client.post("/generate", json={})
    assert response.status_code == 422
    assert "detail" in response.json() # Pydantic validation error

def test_generate_failure_from_graph(client: TestClient, mocker):
    mock_func = mocker.patch("api.generate_codebase", new_callable=AsyncMock)
    mock_func.return_value = {"success": False, "error": "Custom error", "details": "Custom details"}
    
    response = client.post("/generate", json={"prompt": "build an app"})
    assert response.status_code == 422
    js = response.json()
    assert js["success"] is False
    assert js["error"] == "Custom error"
    assert js["details"] == "Custom details"

def test_generate_missing_coder_state(client: TestClient, mocker):
    mock_func = mocker.patch("api.generate_codebase", new_callable=AsyncMock)
    mock_func.return_value = {"success": True} # Missing coder_state
    
    response = client.post("/generate", json={"prompt": "build an app"})
    assert response.status_code == 422
    js = response.json()
    assert js["error"] == "Missing coder state"

def test_generate_success(client: TestClient, mocker):
    mock_func = mocker.patch("api.generate_codebase", new_callable=AsyncMock)
    mock_files = [{"path": "index.html", "content": "<h1>Hello</h1>"}]
    mock_func.return_value = {"success": True, "coder_state": {"generated_files": mock_files}}
    
    response = client.post("/generate", json={"prompt": "build an app"})
    assert response.status_code == 200
    js = response.json()
    assert js["success"] is True
    assert js["files"] == mock_files

def test_generate_exception(client: TestClient, mocker):
    mock_func = mocker.patch("api.generate_codebase", new_callable=AsyncMock)
    mock_func.side_effect = Exception("Database crash")
    
    response = client.post("/generate", json={"prompt": "build an app"})
    assert response.status_code == 500
    js = response.json()
    assert js["success"] is False
    assert "Database crash" in js["details"]
