# Sample tourism data for Jharkhand attractions
import json

ATTRACTIONS_DATA = {
    "Ranchi": [
        {
            "id": "ranchi_001",
            "name": "Hundru Falls",
            "city": "Ranchi",
            "coordinates": {"lat": 23.4230, "lng": 85.5979},
            "type": "waterfall",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "A spectacular waterfall with a drop of 98 meters, perfect for adventure enthusiasts.",
            "best_time": "October to March",
            "duration": "2-3 hours",
            "image": "hundru_falls.jpg"
        },
        {
            "id": "ranchi_002",
            "name": "Dassam Falls",
            "city": "Ranchi",
            "coordinates": {"lat": 23.4657, "lng": 85.4126},
            "type": "waterfall",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Beautiful waterfall formed by Kanchi River, ideal for picnics.",
            "best_time": "October to March",
            "duration": "2-3 hours",
            "image": "dassam_falls.jpg"
        },
        {
            "id": "ranchi_003",
            "name": "Jonha Falls",
            "city": "Ranchi",
            "coordinates": {"lat": 23.5149, "lng": 85.5312},
            "type": "waterfall",
            "interest_tags": ["Adventure", "Spirituality", "Relaxation"],
            "description": "Also known as Gaurav Falls, this place has mythological significance.",
            "best_time": "October to March",
            "duration": "2-3 hours",
            "image": "jonha_falls.jpg"
        },
        {
            "id": "ranchi_004",
            "name": "Panch Gagh Falls",
            "city": "Ranchi",
            "coordinates": {"lat": 23.4867, "lng": 85.4521},
            "type": "waterfall",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Five streams merging to form a beautiful waterfall.",
            "best_time": "October to March",
            "duration": "3-4 hours",
            "image": "panch_gagh_falls.jpg"
        },
        {
            "id": "ranchi_005",
            "name": "Birsa Zoological Park",
            "city": "Ranchi",
            "coordinates": {"lat": 23.4101, "lng": 85.4399},
            "type": "zoo",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Home to various wildlife species including tigers, leopards, and elephants.",
            "best_time": "October to March",
            "duration": "3-4 hours",
            "image": "birsa_zoo.jpg"
        },
        {
            "id": "ranchi_006",
            "name": "Ranchi Lake",
            "city": "Ranchi",
            "coordinates": {"lat": 23.3441, "lng": 85.3096},
            "type": "lake",
            "interest_tags": ["Relaxation"],
            "description": "Artificial lake perfect for boating and evening walks.",
            "best_time": "Year round",
            "duration": "1-2 hours",
            "image": "ranchi_lake.jpg"
        },
        {
            "id": "ranchi_007",
            "name": "Jagannath Temple",
            "city": "Ranchi",
            "coordinates": {"lat": 23.3569, "lng": 85.3350},
            "type": "temple",
            "interest_tags": ["Spirituality", "Culture"],
            "description": "Ancient temple dedicated to Lord Jagannath with beautiful architecture.",
            "best_time": "Year round",
            "duration": "1-2 hours",
            "image": "jagannath_temple.jpg"
        },
        {
            "id": "ranchi_008",
            "name": "Pahari Mandir",
            "city": "Ranchi",
            "coordinates": {"lat": 23.3775, "lng": 85.3352},
            "type": "temple",
            "interest_tags": ["Spirituality", "Adventure"],
            "description": "Hilltop temple offering panoramic views of Ranchi city.",
            "best_time": "Year round",
            "duration": "2-3 hours",
            "image": "pahari_mandir.jpg"
        },
                {
  "id": "ranchi_009",
  "name": "Hathidari Underground Coal Mine",
  "city": "Ranchi",
  "coordinates": {"lat": 23.651219, "lng": 85.358799},
  "type": "tour",
  "interest_tags": ["Spirituality", "Nature", "Exploration"],
  "description": "A one-day guided mining tour blending spiritual insights with Ranchi’s lush landscapes and waterfalls.",
  "best_time": "October to March",
  "duration": "1 Day",
  "group_size": "20-25 Guests",
  "price": "₹2800/person",
  "image": "religious_mining_tour.jpg"
},
    ],
    "Jamshedpur": [
        {
            "id": "jamshedpur_001",
            "name": "Dalma Hills",
            "city": "Jamshedpur",
            "coordinates": {"lat": 22.8465, "lng": 86.1032},
            "type": "hill",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Wildlife sanctuary with trekking trails and elephant sightings.",
            "best_time": "October to March",
            "duration": "4-6 hours",
            "image": "dalma_hills.jpg"
        },
        {
            "id": "jamshedpur_002",
            "name": "Jubilee Park",
            "city": "Jamshedpur",
            "coordinates": {"lat": 22.8046, "lng": 86.2029},
            "type": "park",
            "interest_tags": ["Relaxation", "Culture"],
            "description": "One of the largest parks in Asia with gardens, lake, and zoo.",
            "best_time": "October to March",
            "duration": "3-4 hours",
            "image": "jubilee_park.jpg"
        },
        {
            "id": "jamshedpur_003",
            "name": "Tata Steel Zoological Park",
            "city": "Jamshedpur",
            "coordinates": {"lat": 22.7980, "lng": 86.2027},
            "type": "zoo",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Modern zoo with diverse wildlife and conservation programs.",
            "best_time": "October to March",
            "duration": "3-4 hours",
            "image": "tata_zoo.jpg"
        },
        {
            "id": "jamshedpur_004",
            "name": "Bhuvneshwari Temple",
            "city": "Jamshedpur",
            "coordinates": {"lat": 22.8567, "lng": 86.1678},
            "type": "temple",
            "interest_tags": ["Spirituality", "Culture"],
            "description": "Sacred temple dedicated to Goddess Bhuvneshwari.",
            "best_time": "Year round",
            "duration": "1-2 hours",
            "image": "bhuvneshwari_temple.jpg"
        }
    ],
    "Hazaribagh": [
        {
            "id": "hazaribagh_001",
            "name": "Canary Hill",
            "city": "Hazaribagh",
            "coordinates": {"lat": 23.9929, "lng": 85.3644},
            "type": "hill",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Scenic hilltop with panoramic views and sunset points.",
            "best_time": "October to March",
            "duration": "2-3 hours",
            "image": "canary_hill.jpg"
        },
        {
            "id": "hazaribagh_002",
            "name": "Hazaribagh Wildlife Sanctuary",
            "city": "Hazaribagh",
            "coordinates": {"lat": 23.9441, "lng": 85.2734},
            "type": "sanctuary",
            "interest_tags": ["Adventure", "Relaxation"],
            "description": "Rich biodiversity with tigers, leopards, and various bird species.",
            "best_time": "October to March",
            "duration": "4-6 hours",
            "image": "hazaribagh_sanctuary.jpg"
        },
        {
            "id": "hazaribagh_003",
            "name": "Suryakund",
            "city": "Hazaribagh",
            "coordinates": {"lat": 24.0123, "lng": 85.3987},
            "type": "hot_spring",
            "interest_tags": ["Spirituality", "Relaxation"],
            "description": "Natural hot springs with medicinal properties.",
            "best_time": "October to March",
            "duration": "2-3 hours",
            "image": "suryakund.jpg"
        }
    ],
    "Sahibganj": [
        {
            "id": "sahibganj_001",
            "name": "Rajmahal",
            "city": "Sahibganj",
            "coordinates": {"lat": 25.0504, "lng": 87.8314},
            "type": "historical",
            "interest_tags": ["Culture", "Spirituality"],
            "description": "Historical town with ancient ruins and archaeological significance.",
            "best_time": "October to March",
            "duration": "3-4 hours",
            "image": "rajmahal.jpg"
        },
        {
            "id": "sahibganj_002",
            "name": "Teliagarhi Fort",
            "city": "Sahibganj",
            "coordinates": {"lat": 25.2534, "lng": 87.6234},
            "type": "fort",
            "interest_tags": ["Culture", "Adventure"],
            "description": "Ancient fort with historical significance and architectural beauty.",
            "best_time": "October to March",
            "duration": "2-3 hours",
            "image": "teliagarhi_fort.jpg"
        }
    ],
    "Dhanbad": [
        {
            "id": "dhanbad_001",
            "name": "Dalmi Temple",
            "city": "Dhanbad",
            "coordinates": {"lat": 23.7957, "lng": 86.4304},
            "type": "temple",
            "interest_tags": ["Spirituality", "Culture"],
            "description": "Ancient temple complex with intricate carvings.",
            "best_time": "Year round",
            "duration": "1-2 hours",
            "image": "dalmi_temple.jpg"
        }
    ]
}

def get_all_attractions():
    """Get all attractions as a flat list"""
    all_attractions = []
    for city, attractions in ATTRACTIONS_DATA.items():
        all_attractions.extend(attractions)
    return all_attractions

def get_attractions_by_city(city_name):
    """Get attractions filtered by city"""
    return ATTRACTIONS_DATA.get(city_name, [])

def get_attractions_by_interest(interest):
    """Get attractions filtered by interest tag"""
    filtered_attractions = []
    for city, attractions in ATTRACTIONS_DATA.items():
        for attraction in attractions:
            if interest in attraction.get('interest_tags', []):
                filtered_attractions.append(attraction)
    return filtered_attractions

def get_attraction_by_id(attraction_id):
    """Get specific attraction by ID"""
    for city, attractions in ATTRACTIONS_DATA.items():
        for attraction in attractions:
            if attraction['id'] == attraction_id:
                return attraction
    return None