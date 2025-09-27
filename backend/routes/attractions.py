
from flask import Blueprint, request, jsonify
from data.attractions_data import get_attractions_by_city

AttractionsBlueprint = Blueprint('attractions', __name__)

@AttractionsBlueprint.route('/', methods=['GET'])
def get_city_attractions():
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    return jsonify(get_attractions_by_city(city))
