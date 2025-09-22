import requests
import math
import os
from data.attractions_data import get_all_attractions

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

def geocode_address(address):
    """Convert address to coordinates using Google Maps Geocoding API"""
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        # For demo purposes, return mock coordinates for common locations
        mock_locations = {
            'ranchi': {'lat': 23.3441, 'lng': 85.3096},
            'jamshedpur': {'lat': 22.8046, 'lng': 86.2029},
            'hazaribagh': {'lat': 23.9929, 'lng': 85.3644},
            'sahibganj': {'lat': 25.0504, 'lng': 87.8314},
            'dhanbad': {'lat': 23.7957, 'lng': 86.4304},
            'kolkata': {'lat': 22.5726, 'lng': 88.3639},
            'patna': {'lat': 25.5941, 'lng': 85.1376}
        }
        # Try to find a match in mock locations
        address_lower = address.lower()
        for location, coords in mock_locations.items():
            if location in address_lower:
                return coords

        # Default to Ranchi if no match found
        return {'lat': 23.3441, 'lng': 85.3096}
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == 'OK' and data['results']:
            location = data['results'][0]['geometry']['location']
            return {'lat': location['lat'], 'lng': location['lng']}
        else:
            raise Exception(f"Geocoding failed: {data['status']}")
    
    except Exception as e:
        print(f"Geocoding error: {e}")
        # Return default coordinates (Ranchi) on error
        return {'lat': 23.3441, 'lng': 85.3096}

def calculate_route(origin, destination):
    """Calculate route between origin and destination"""
    # Convert addresses to coordinates
    origin_coords = geocode_address(origin) if isinstance(origin, str) else origin
    dest_coords = geocode_address(destination) if isinstance(destination, str) else destination
    
    # Calculate basic route info
    distance = calculate_distance(
        origin_coords['lat'], origin_coords['lng'],
        dest_coords['lat'], dest_coords['lng']
    )
    
    # Generate waypoints along the route (simplified)
    waypoints = generate_waypoints(origin_coords, dest_coords)
    
    return {
        'origin': origin_coords,
        'destination': dest_coords,
        'distance_km': round(distance, 2),
        'estimated_duration_hours': round(distance / 60, 1),  # Assuming 60 km/h average speed
        'waypoints': waypoints
    }

def generate_waypoints(origin, destination, num_points=5):
    """Generate waypoints along the route"""
    waypoints = []
    
    for i in range(num_points + 1):
        ratio = i / num_points
        lat = origin['lat'] + (destination['lat'] - origin['lat']) * ratio
        lng = origin['lng'] + (destination['lng'] - origin['lng']) * ratio
        waypoints.append({'lat': lat, 'lng': lng})
    
    return waypoints

def find_nearby_attractions(waypoints, buffer_km=10, interests=None):
    """Find attractions near the route waypoints"""
    all_attractions = get_all_attractions()
    nearby_attractions = []
    
    for waypoint in waypoints:
        for attraction in all_attractions:
            # Calculate distance from waypoint to attraction
            distance = calculate_distance(
                waypoint['lat'], waypoint['lng'],
                attraction['coordinates']['lat'], attraction['coordinates']['lng']
            )
            
            # Check if attraction is within buffer distance
            if distance <= buffer_km:
                # Filter by interests if provided
                if interests:
                    attraction_interests = attraction.get('interest_tags', [])
                    if not any(interest in attraction_interests for interest in interests):
                        continue
                
                # Add distance info to attraction
                attraction_with_distance = attraction.copy()
                attraction_with_distance['distance_from_route'] = round(distance, 2)
                
                # Avoid duplicates
                if not any(na['id'] == attraction['id'] for na in nearby_attractions):
                    nearby_attractions.append(attraction_with_distance)
    
    # Sort by distance from route
    nearby_attractions.sort(key=lambda x: x['distance_from_route'])
    
    return nearby_attractions

def optimize_attraction_order(attractions, start_location):
    """Optimize the order of attractions to minimize travel distance"""
    if not attractions:
        return []
    
    optimized = []
    remaining = attractions.copy()
    current_location = start_location
    
    while remaining:
        # Find the nearest unvisited attraction
        nearest = min(remaining, key=lambda a: calculate_distance(
            current_location['lat'], current_location['lng'],
            a['coordinates']['lat'], a['coordinates']['lng']
        ))
        
        optimized.append(nearest)
        remaining.remove(nearest)
        current_location = nearest['coordinates']
    
    return optimized