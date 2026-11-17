
import os
import requests


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://10.22.39.192:11434"
)


def test_ollama_server_is_available():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=15
    )

    assert response.status_code == 200
