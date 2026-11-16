
import os
import base64
import requests

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://10.22.39.192:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5vl:latest"
)


def analyze_property_image(image_path, question):

    # STEP 1: Check whether image exists

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # STEP 2: Read image and convert to Base64

    with open(image_path, "rb") as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    # STEP 3: Prepare the prompt

    prompt = f"""
You are a real estate visual analysis assistant.

Analyse the supplied property image.

User question:
{question}

Instructions:

1. Describe only what is visually observable.
2. Identify visible rooms, fixtures and features.
3. Mention the apparent condition where possible,
   but do not make definitive structural assessments.
4. Do not invent dimensions or property specifications.
5. If something cannot be determined from the image,
   clearly say so.
6. Distinguish visual observations from assumptions.
"""

    # STEP 4: Send image and prompt to Ollama

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",

        json={
            "model": OLLAMA_MODEL,

            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }
            ],

            "stream": False,

            "options": {
                "temperature": 0
            }
        },

        timeout=180
    )

    response.raise_for_status()

    # STEP 5: Extract Qwen's response

    result = response.json()

    return result["message"]["content"]
