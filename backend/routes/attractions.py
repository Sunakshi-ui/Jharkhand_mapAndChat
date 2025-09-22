# from flask import Blueprint, jsonify, request
# from data.attractions_data import get_all_attractions, get_attractions_by_city, get_attractions_by_interest

# AttractionsBlueprint = Blueprint('attractions', __name__)

# @AttractionsBlueprint.route('/', methods=['GET'])
# def get_attractions():
#     """Get all attractions or filter by city/interest"""
#     city = request.args.get('city')
#     interest = request.args.get('interest')
    
#     if city:
#         attractions = get_attractions_by_city(city)
#     elif interest:
#         attractions = get_attractions_by_interest(interest)
#     else:
#         attractions = get_all_attractions()
    
#     return jsonify({
#         'success': True,
#         'data': attractions,
#         'count': len(attractions)
#     })

# @AttractionsBlueprint.route('/cities', methods=['GET'])
# def get_cities():
#     """Get list of available cities"""
#     cities = ['Ranchi', 'Jamshedpur', 'Hazaribagh', 'Sahibganj', 'Dhanbad']
#     return jsonify({
#         'success': True,
#         'data': cities
#     })

# @AttractionsBlueprint.route('/interests', methods=['GET'])
# def get_interests():
#     """Get list of available interest categories"""
#     interests = ['Adventure', 'Culture', 'Food', 'Spirituality', 'Relaxation']
#     return jsonify({
#         'success': True,
#         'data': interests
#     })

# routes/attractions.py
from flask import Blueprint, request, jsonify
from data.attractions_data import get_attractions_by_city

AttractionsBlueprint = Blueprint('attractions', __name__)

@AttractionsBlueprint.route('/', methods=['GET'])
def get_city_attractions():
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    return jsonify(get_attractions_by_city(city))
