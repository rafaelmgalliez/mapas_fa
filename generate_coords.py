import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Load your original CSV file
try:
    df = pd.read_csv("fa_casoshumanos_1994-2024.csv", sep=';', encoding='utf-8', on_bad_lines='warn')
except UnicodeDecodeError:
    df = pd.read_csv("fa_casoshumanos_1994-2024.csv", sep=';', encoding='latin-1', on_bad_lines='warn')

# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="febre_amarela_geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) # Be mindful of rate limits

def get_coordinates(municipio, estado):
    """Geocodes a municipality and returns latitude and longitude."""
    try:
        location = geocode(f"{municipio}, {estado}, Brasil")
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding {municipio}, {estado}: {e}")
        return None, None

# Create new columns for latitude and longitude
df['geocoded_latitude'] = None
df['geocoded_longitude'] = None

# Geocode each unique municipality and store the coordinates
unique_municipalities = df[['MUN_LPI', 'UF_LPI']].drop_duplicates()
municipality_coordinates_cache = {}

for index, row in unique_municipalities.iterrows():
    municipio = row['MUN_LPI']
    estado = row['UF_LPI']
    if (municipio, estado) not in municipality_coordinates_cache:
        latitude, longitude = get_coordinates(municipio, estado)
        municipality_coordinates_cache[(municipio, estado)] = (latitude, longitude)
        time.sleep(1.1) # Respect rate limits

# Map the geocoded coordinates back to the original DataFrame
def map_coordinates(row):
    coords = municipality_coordinates_cache.get((row['MUN_LPI'], row['UF_LPI']))
    if coords:
        return coords[0], coords[1]
    return None, None

df[['geocoded_latitude', 'geocoded_longitude']] = df.apply(map_coordinates, axis=1, result_type='expand')

# Save the DataFrame with geocoded coordinates to a new CSV file
df.to_csv("FA_cases_geocoded.csv", sep=';', encoding='utf-8', index=False)

print("Geocoding complete. Results saved to FA_cases_geocoded.csv")
