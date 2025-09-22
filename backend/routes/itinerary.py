from flask import Blueprint, jsonify, request
import json
import uuid
from datetime import datetime

ItineraryBlueprint = Blueprint('itinerary', __name__)

# In-memory storage for demo purposes (replace with database in production)
itineraries = {}

@ItineraryBlueprint.route('/', methods=['POST'])
def create_itinerary():
    """Create a new itinerary"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    # Generate unique ID
    itinerary_id = str(uuid.uuid4())
    
    # Create itinerary object
    itinerary = {
        'id': itinerary_id,
        'user_id': data.get('user_id'),
        'title': data.get('title', 'My Jharkhand Trip'),
        'start_date': data.get('start_date'),
        'end_date': data.get('end_date'),
        'current_location': data.get('current_location'),
        'interests': data.get('interests', []),
        'days': data.get('days', []),
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    itineraries[itinerary_id] = itinerary
    
    return jsonify({
        'success': True,
        'data': itinerary,
        'share_link': f'/itinerary/{itinerary_id}'
    }), 201

@ItineraryBlueprint.route('/<string:itinerary_id>', methods=['GET'])
def get_itinerary(itinerary_id):
    """Get itinerary by ID"""
    if itinerary_id not in itineraries:
        return jsonify({'success': False, 'message': 'Itinerary not found'}), 404
    
    return jsonify({
        'success': True,
        'data': itineraries[itinerary_id]
    })

@ItineraryBlueprint.route('/<string:itinerary_id>', methods=['PUT'])
def update_itinerary(itinerary_id):
    """Update an existing itinerary"""
    if itinerary_id not in itineraries:
        return jsonify({'success': False, 'message': 'Itinerary not found'}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    # Update itinerary
    itinerary = itineraries[itinerary_id]
    for key, value in data.items():
        if key != 'id':  # Prevent ID modification
            itinerary[key] = value
    
    itinerary['updated_at'] = datetime.now().isoformat()
    
    return jsonify({
        'success': True,
        'data': itinerary
    })

@ItineraryBlueprint.route('/<string:itinerary_id>/export', methods=['GET'])
def export_itinerary(itinerary_id):
    """Export itinerary as JSON"""
    if itinerary_id not in itineraries:
        return jsonify({'success': False, 'message': 'Itinerary not found'}), 404
    
    export_format = request.args.get('format', 'json')
    
    if export_format == 'json':
        return jsonify({
            'success': True,
            'data': itineraries[itinerary_id],
            'format': 'json'
        })
    
    # TODO: Implement PDF export
    return jsonify({'success': False, 'message': 'Format not supported yet'}), 400