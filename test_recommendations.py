
from recommendations import recommend_properties


def test_recommend_properties():
    results = recommend_properties(
        budget=20_000_000,
        preferred_location=None,
        bedrooms=2,
        top_n=3
    )

    assert results is not None
    assert len(results) <= 3

    if not results.empty:
        assert "property_id" in results.columns
        assert "recommendation_score" in results.columns
        assert (results["price"] <= 20_000_000).all()
        assert (results["bedrooms"] == 2).all()
