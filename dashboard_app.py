import pandas as pd
import plotly.express as px
import streamlit as st

from app import setup_database
from dashboard import (
    filter_properties,
    get_market_statistics,
    get_properties
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mumbai Property Intelligence",
    page_icon="🏢",
    layout="wide"
)


# ==================================================
# HELPERS
# ==================================================

def format_number(value, suffix=""):
    """Format a number with thousands separators, or 'N/A' if missing."""

    if pd.isna(value):
        return "N/A"

    return f"{int(value):,}{suffix}"


# ==================================================
# DATABASE
# ==================================================

@st.cache_resource
def init_database():
    """Create the property database once per server process."""

    setup_database()
    return True


init_database()

df = get_properties()

if df.empty:
    st.title("🏢 Mumbai Property Intelligence")
    st.warning("The property database is empty.")
    st.stop()

# The database column is `property_id`; fall back gracefully if
# get_properties() renames it.
if "property_id" in df.columns:
    ID_COLUMN = "property_id"
elif "id" in df.columns:
    ID_COLUMN = "id"
else:
    ID_COLUMN = None


# ==================================================
# HEADER
# ==================================================

st.title("🏢 Mumbai Property Intelligence")

st.caption("Real estate listings, pricing and market insights")


# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.header("Property Filters")

locations = ["All"] + sorted(
    df["location"].dropna().unique().tolist()
)

property_types = ["All"] + sorted(
    df["property_type"].dropna().unique().tolist()
)

location = st.sidebar.selectbox("Location", locations)

property_type = st.sidebar.selectbox("Property Type", property_types)

bedrooms = st.sidebar.selectbox("Bedrooms", ["All", 1, 2, 3, 4])

max_price = st.sidebar.number_input(
    "Maximum Budget (₹)",
    min_value=0,
    value=50000000,
    step=500000
)


# ==================================================
# FILTER DATA
# ==================================================

filtered_df = filter_properties(
    df,
    location=location,
    property_type=property_type,
    bedrooms=None if bedrooms == "All" else bedrooms,
    max_price=max_price
)


# ==================================================
# MARKET STATISTICS
# ==================================================

stats = get_market_statistics(filtered_df)

st.subheader("Market Overview")

if stats:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Available Properties", stats["total_properties"])

    col2.metric("Average Price", f"₹{stats['average_price']:,.0f}")

    col3.metric("Minimum Price", f"₹{stats['minimum_price']:,.0f}")

    col4.metric("Average Area", f"{stats['average_area']:,.0f} sqft")

else:

    st.info("No properties match your filters.")


# ==================================================
# PROPERTY LISTINGS
# ==================================================

st.divider()

st.subheader("Available Properties")

if not filtered_df.empty:

    for index, property_row in filtered_df.iterrows():

        property_number = (
            property_row[ID_COLUMN] if ID_COLUMN else index
        )

        with st.container(border=True):

            col1, col2 = st.columns([3, 1])

            with col1:

                st.markdown(f"### Property #{property_number}")

                st.write(f"📍 {property_row['location']}")

                st.write(f"Type: {property_row['property_type']}")

                # Commercial listings have no bedroom count
                if "bedrooms" in filtered_df.columns:
                    st.write(
                        f"Bedrooms: {format_number(property_row['bedrooms'])}"
                    )

                if "area_sqft" in filtered_df.columns:
                    st.write(
                        f"Area: {format_number(property_row['area_sqft'], ' sqft')}"
                    )

            with col2:

                st.metric("Price", f"₹{property_row['price']:,.0f}")

else:

    st.warning("No matching properties found.")


# ==================================================
# PRICE CHART
# ==================================================

st.divider()

st.subheader("Property Price Comparison")

if not filtered_df.empty:

    chart = px.bar(
        filtered_df,
        x="location",
        y="price",
        color="property_type",
        hover_data=[ID_COLUMN] if ID_COLUMN else None,
        title="Property prices by location"
    )

    st.plotly_chart(chart, use_container_width=True)


# ==================================================
# LOCATION ANALYSIS
# ==================================================

st.divider()

st.subheader("Properties by Location")

if not filtered_df.empty:

    location_counts = (
        filtered_df.groupby("location")
        .size()
        .reset_index(name="property_count")
    )

    pie_chart = px.pie(
        location_counts,
        names="location",
        values="property_count",
        title="Property distribution"
    )

    st.plotly_chart(pie_chart, use_container_width=True)