import folium
from geopy.geocoders import Nominatim
from ner_extractor import extract_location  # Import location extractor

# Initialize geocoder
geolocator = Nominatim(user_agent="tweet_visualizer")

def plot_locations_on_map(locations):
    """Plot extracted locations on a folium map."""
    # Initialize map centered at a default location (e.g., New York)
    map_center = [40.7128, -74.0060]  # New York coordinates
    tweet_map = folium.Map(location=map_center, zoom_start=12)

    for location in locations:
        try:
            # Geocode location to get latitude and longitude
            location_obj = geolocator.geocode(location)
            if location_obj:
                folium.Marker(
                    location=[location_obj.latitude, location_obj.longitude],
                    popup=location
                ).add_to(tweet_map)
        except Exception as e:
            print(f"Error geocoding location {location}: {e}")
    
    # Save the map as an HTML file
    tweet_map.save("locations_map.html")
    print("Map has been saved as 'locations_map.html'.")

# Test visualization with some example locations
if __name__ == "__main__":
    # Example locations from negative tweets
    test_locations = ["New York", "San Francisco", "Chicago"]
    plot_locations_on_map(test_locations)
