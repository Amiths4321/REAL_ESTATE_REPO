
import sqlite3
import pandas as pd

DB_NAME = "real_estate_ai.db"


def get_properties():

    connection = sqlite3.connect(DB_NAME)

    query = """
        SELECT *
        FROM properties
        WHERE status = 'Available'
    """

    df = pd.read_sql_query(query, connection)

    connection.close()

    return df


def filter_properties(
    df,
    location=None,
    property_type=None,
    bedrooms=None,
    max_price=None
):

    filtered = df.copy()

    if location and location != "All":

        filtered = filtered[
            filtered["location"] == location
        ]

    if property_type and property_type != "All":

        filtered = filtered[
            filtered["property_type"] == property_type
        ]

    if bedrooms is not None:

        if "bedrooms" in filtered.columns:
            filtered = filtered[
                filtered["bedrooms"] == bedrooms
            ]

    if max_price is not None:

        filtered = filtered[
            filtered["price"] <= max_price
        ]

    return filtered


def get_market_statistics(df):

    if df.empty:
        return {}

    return {
        "total_properties": len(df),
        "average_price": df["price"].mean(),
        "minimum_price": df["price"].min(),
        "maximum_price": df["price"].max(),
        "average_area": (
            df["area_sqft"].mean()
            if "area_sqft" in df.columns
            else 0
        )
    }
