
from recommendations import recommend_properties

results = recommend_properties(
    budget=20000000,
    preferred_location=None,
    bedrooms=2,
    top_n=3
)

if results.empty:

    print("No matching properties found.")

else:

    print("\nTOP PROPERTY RECOMMENDATIONS\n")

    for _, property in results.iterrows():

        print("Property ID:", property["property_id"])
        print("Location:", property["location"])
        print("Price:", f"₹{property['price']:,.0f}")
        print("Area:", property["area_sqft"], "sqft")
        print(
            "Price per sqft:",
            f"₹{property['price_per_sqft']:,.0f}"
        )
        print(
            "Recommendation score:",
            round(property["recommendation_score"], 2)
        )

        print("-" * 40)
