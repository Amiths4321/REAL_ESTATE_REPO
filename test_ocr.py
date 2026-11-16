from ollama import Client

client = Client(host="http://10.22.39.192:11434")

response = client.chat(
    model="glm-ocr:latest",
    messages=[
        {
            "role": "user",
            "content": "Extract all text from this image.",
            "images": ["sample.jpg"],  # Replace with your actual image path
        }
    ],
)

print(response["message"]["content"])