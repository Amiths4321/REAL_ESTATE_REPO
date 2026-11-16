
from vision import analyze_property_image

image_path = "images/apartment.jpg"

question = """
Describe this apartment.

Identify:
1. Visible rooms and furniture.
2. Flooring and wall finishes.
3. Visible lighting and windows.
4. Any visible amenities or fixtures.
5. Overall visual condition.
"""

answer = analyze_property_image(
    image_path,
    question
)

print("\nPROPERTY IMAGE ANALYSIS\n")
print(answer)
