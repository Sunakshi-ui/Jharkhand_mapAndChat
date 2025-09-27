import json

hotels_data = {

  "Ranchi": [
    {
      "id": "latehar_001",
      "name": "Prabhat Vihar Deluxe, Netarhat",
      "city": "Latehar",
      "coordinates": {"lat": 23.473776, "lng": 84.279593},
      "rooms": 39,
      "contact": "9102403883",
      "amenities": ["Wash room", "Parking"],
      "price_range": "₹1000-5000"
    },
    
  {
    "id": "latehar_002",
    "name": "Hotel Van Vihar, Betla",
    "city": "Latehar",
    "coordinates": {"lat": 23.886997, "lng": 84.191706},
    "rooms": 23,
    "contact": "9102403882",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹1000-1700"
  },
  {
    "id": "ramgarh_001",
    "name": "Sarovar Vihar, Patratu Lake Resort",
    "city": "Ramgarh",
    "coordinates": {"lat": 23.610083, "lng": 85.281591},
    "rooms": 20,
    "contact": "9905900149",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹2300-6750"
  },
  {
    "id": "ramgarh_002",
    "name": "Paryatan Vihar Patratu",
    "city": "Ramgarh",
    "coordinates": {"lat": 23.609404, "lng": 85.280906},
    "rooms": 20,
    "contact": "9905900149",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹3500-6000"
  },
  {
    "id": "ranchi_004",
    "name": "Hotel Birsa Vihar, Ranchi",
    "city": "Ranchi",
    "coordinates": {"lat": 23.352191, "lng": 85.324694},
    "rooms": 26,
    "contact": "7632987037",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹2000-3500"
  }
  ],

   "Deoghar": [
  {
    "id": "deoghar_001",
    "name": "Hotel Natraj Vihar, Deoghar",
    "city": "Deoghar",
    "coordinates": {"lat": 24.488577, "lng": 86.698604},
    "rooms": 20,
    "contact": "9102403877",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹1000-3000"
  },
  {
    "id": "deoghar_002",
    "name": "Hotel Baidyanath Vihar, Deoghar",
    "city": "Deoghar",
    "coordinates": {"lat": 24.488945, "lng": 86.696191},
    "rooms": 23,
    "contact": "7091591307",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹300-750"
  }
   ],

   "Dhanbad": [
  {
    "id": "dhanbad_001",
    "name": "Hotel Ratan Vihar, Dhanbad",
    "city": "Dhanbad",
    "coordinates": {"lat": 23.795865, "lng": 86.433702},
    "rooms": 16,
    "contact": "9102403878",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹700-3000"
  },
  {
    "id": "dumka_001",
    "name": "Hotel Basuki Vihar, Basukinath",
    "city": "Dumka",
    "coordinates": {"lat": 24.395853, "lng": 87.086295},
    "rooms": 11,
    "contact": "9102403876",
    "amenities": ["Wash room", "Parking"],
    "price_range": "₹800-1200"
  }
]
}


# Helper functions for hotels
def get_all_hotels():
  """Get all hotels as a flat list"""
  all_hotels = []
  for city, hotels in hotels_data.items():
    all_hotels.extend(hotels)
  return all_hotels

def get_hotels_by_city(city_name):
  """Get hotels filtered by city"""
  return hotels_data.get(city_name, [])

def get_hotel_by_id(hotel_id):
  """Get specific hotel by ID"""
  for city, hotels in hotels_data.items():
    for hotel in hotels:
      if hotel['id'] == hotel_id:
        return hotel
  return None