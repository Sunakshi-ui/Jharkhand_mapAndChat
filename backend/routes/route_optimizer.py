from flask import Blueprint, jsonify, request
from utils.route_calculator import calculate_route, find_nearby_attractions
from data.attractions_data import get_all_attractions

RouteOptimizerBlueprint = Blueprint('route_optimizer', __name__)

@RouteOptimizerBlueprint.route('/calculate', methods=['POST'])
def calculate_route_with_suggestions():
    """Calculate route between two points and suggest nearby attractions"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    origin = data.get('origin')
    destination = data.get('destination')
    interests = data.get('interests', [])
    buffer_km = data.get('buffer_km', 10)
    
    if not origin or not destination:
        return jsonify({
            'success': False, 
            'message': 'Origin and destination are required'
        }), 400
    
    try:
        # Calculate basic route
        route_info = calculate_route(origin, destination)
        
        # Find nearby attractions along the route
        nearby_attractions = find_nearby_attractions(
            route_info['waypoints'], 
            buffer_km, 
            interests
        )
        
        return jsonify({
            'success': True,
            'data': {
                'route': route_info,
                'nearby_attractions': nearby_attractions,
                'suggestions_count': len(nearby_attractions)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Route calculation failed: {str(e)}'
        }), 500

@RouteOptimizerBlueprint.route('/optimize', methods=['POST'])
def optimize_itinerary():
    """Optimize itinerary based on location proximity and interests"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    attractions = data.get('attractions', [])
    interests = data.get('interests', [])
    start_location = data.get('start_location')
    
    if not attractions:
        return jsonify({
            'success': False,
            'message': 'Attractions list is required'
        }), 400
    
    # TODO: Implement optimization algorithm
    # For now, return a simple grouped optimization
    optimized_days = []
    
    # Group attractions by city for better routing
    city_groups = {}
    for attraction in attractions:
        city = attraction.get('city', 'Unknown')
        if city not in city_groups:
            city_groups[city] = []
        city_groups[city].append(attraction)
    
    day_counter = 1
    for city, city_attractions in city_groups.items():
        optimized_days.append({
            'day': day_counter,
            'city': city,
            'attractions': city_attractions,
            'estimated_duration': len(city_attractions) * 2  # 2 hours per attraction
        })
        day_counter += 1
    
    return jsonify({
        'success': True,
        'data': {
            'optimized_days': optimized_days,
            'total_days': len(optimized_days),
            'total_attractions': len(attractions)
        }
    })