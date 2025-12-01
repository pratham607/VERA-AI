import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

print("🏙️ Welcome to the Multi-City Housing Price Prediction System 🏡")
print("Cities available: Chicago, New York, California")
city = input("\nEnter city name: ").strip().lower()

# =========================
#  CHICAGO MODEL
# =========================
if city == "chicago":
    print("\n📂 Loading Chicago dataset...")
    df = pd.read_csv("cleaned_chicago_file2.csv")

    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"].astype(str))
    df["year_built_encoded"] = le.fit_transform(df["year_built"].astype(str))
    df["status_encoded"] = le.fit_transform(df["status"].astype(str))
    df["soldOn_encoded"] = le.fit_transform(df["soldOn"].astype(str))

    features = [
        "type_encoded", "year_built_encoded", "beds", "baths",
        "baths_full", "garage", "lot_sqft", "sqft",
        "stories", "lastSoldPrice", "soldOn_encoded", "status_encoded"
    ]
    X = df[features]
    y = df["listPrice"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Chicago model trained successfully!")

    print("\nPlease enter the details of the house:")
    user_input = {
        "type_encoded": 0,
        "year_built_encoded": int(input("Year built: ")),
        "beds": int(input("Number of beds: ")),
        "baths": float(input("Number of baths: ")),
        "baths_full": float(input("Full baths: ")),
        "garage": int(input("Garage spaces: ")),
        "lot_sqft": float(input("Lot size (sqft): ")),
        "sqft": float(input("Square feet: ")),
        "stories": int(input("Number of stories: ")),
        "lastSoldPrice": float(input("Last sold price: ")),
        "soldOn_encoded": 0,
        "status_encoded": 0
    }

# =========================
#  NEW YORK MODEL
# =========================
elif city == "new york":
    print("\n📂 Loading New York dataset...")
    df = pd.read_csv("cleanednewyork housing data.csv")
    df = df[df["SALE PRICE"] > 1000]

    for col in ["LAND SQUARE FEET", "GROSS SQUARE FEET", "SALE PRICE"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

    df.dropna(subset=["LAND SQUARE FEET", "GROSS SQUARE FEET", "SALE PRICE"], inplace=True)

    le = LabelEncoder()
    for col in ["NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "BUILDING CLASS AT PRESENT", "BUILDING CLASS AT TIME OF SALE"]:
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))

    features = [
        "NEIGHBORHOOD_encoded", "BUILDING CLASS CATEGORY_encoded",
        "BUILDING CLASS AT PRESENT_encoded", "BUILDING CLASS AT TIME OF SALE_encoded",
        "ZIP CODE", "RESIDENTIAL UNITS", "LAND SQUARE FEET", "GROSS SQUARE FEET", "YEAR BUILT"
    ]
    X = df[features]
    y = df["SALE PRICE"]

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print("✅ New York model trained successfully!")

    print("\nPlease enter the details of the house:")
    user_input = {
        "NEIGHBORHOOD_encoded": 0,
        "BUILDING CLASS CATEGORY_encoded": 0,
        "BUILDING CLASS AT PRESENT_encoded": 0,
        "BUILDING CLASS AT TIME OF SALE_encoded": 0,
        "ZIP CODE": int(input("ZIP Code: ")),
        "RESIDENTIAL UNITS": int(input("Residential units: ")),
        "LAND SQUARE FEET": float(input("Land sqft: ")),
        "GROSS SQUARE FEET": float(input("Gross sqft: ")),
        "YEAR BUILT": int(input("Year built: "))
    }

# =========================
#  CALIFORNIA MODEL
# =========================
elif city == "california":
    print("\n📂 Loading California dataset...")
    df = pd.read_csv("Cleaned housing data.csv")

    le = LabelEncoder()
    df["ocean_proximity_encoded"] = le.fit_transform(df["ocean_proximity"].astype(str))

    features = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "ocean_proximity_encoded"
    ]
    X = df[features]
    y = df["median_house_value"]

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print("✅ California model trained successfully!")

    print("\nPlease enter the details of the house:")
    user_input = {
        "longitude": float(input("Longitude: ")),
        "latitude": float(input("Latitude: ")),
        "housing_median_age": int(input("House age: ")),
        "total_rooms": int(input("Total rooms: ")),
        "total_bedrooms": int(input("Total bedrooms: ")),
        "population": int(input("Population: ")),
        "households": int(input("Households: ")),
        "median_income": float(input("Median income: ")),
        "ocean_proximity_encoded": 0
    }

# =========================
#  INVALID CITY
# =========================
else:
    print("\n❌ City not recognized. Please choose Chicago, New York, or California.")
    exit()

# =========================
#  PREDICTION
# =========================
user_df = pd.DataFrame([user_input])
predicted_price = model.predict(user_df)[0]

print(f"\n💰 Predicted House Price in {city.title()}: ${predicted_price:,.2f}")
print("------------------------------------------------------")
print("✅ Prediction Complete! Thank you for using our model.")

