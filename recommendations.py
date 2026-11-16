
import sqlite3
import pandas as pd

DB_NAME = "real_estate_ai.db"


def get_available_properties():

    connection = sqlite3.connect(DB_NAME)

    query = """
          SELECT
                    property_id,
                    property_type,
                    location,
                    price,
                    bedrooms,
                    bathrooms,
                    area_sqft,
                    status
          FROM properties
          WHERE status = 'Available'
          """,
    df = pd.read_sql_query(query, connection
    )

    connection.close()

    return df


def calculate_property_score(
    df,
    budget,
    preferred_location=None,
    bedrooms=None
):

    df = df.copy()

    if df.empty:
        return df

    # Calculate price per square foot
    df["price_per_sqft"] = (
        df["price"] / df["area_sqft"]
    )

    # Calculate budget score
    df["budget_score"] = (
        1 - (df["price"] / budget)
    ).clip(lower=0, upper=1) * 100

    # Location score
    if preferred_location:

        df["location_score"] = (
            df["location"].str.lower()
            == preferred_location.lower()
        ).astype(int) * 100

    else:

        df["location_score"] = 50

    # Bedroom score
    if bedrooms is not None:

        df["bedroom_score"] = (
            df["bedrooms"] == bedrooms
        ).astype(int) * 100

    else:

        df["bedroom_score"] = 50

    # Weighted recommendation score
    df["recommendation_score"] = (
        df["budget_score"] * 0.4
        + df["location_score"] * 0.3
        + df["bedroom_score"] * 0.3
    )

    return df.sort_values(
        by="recommendation_score",
        ascending=False
    )

def check_columns():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("PRAGMA table_info(properties)")

    columns = cursor.fetchall()

    for column in columns:
        print(column[1])

    connection.close()


check_columns()

def recommend_properties(
    budget,
    preferred_location=None,
    bedrooms=None,
    top_n=3
):

    df = get_available_properties()

    if df.empty:
        return df

    # Apply hard filters first
    df = df[df["price"] <= budget]

    if preferred_location:

        df = df[
            df["location"].str.lower()
            == preferred_location.lower()
        ]

    if bedrooms is not None:

        df = df[
            df["bedrooms"] == bedrooms
        ]

    if df.empty:
        return df

    df = calculate_property_score(
        df,
        budget,
        preferred_location,
        bedrooms
    )

    return df.head(top_n)
