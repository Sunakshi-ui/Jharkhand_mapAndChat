import google.generativeai as genai
from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

# Import attractions data
from data.attractions_data import get_all_attractions, get_attractions_by_city, get_attractions_by_interest, get_attraction_by_id
from data.hotels_data import get_all_hotels, get_hotels_by_city, get_hotel_by_id

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection with fallback
try:
    mongo_url = os.environ['MONGO_URL']
    client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
    db = client[os.environ['DB_NAME']]
    # Test connection
    import asyncio
    async def test_connection():
        try:
            await client.admin.command('ismaster')
            return True
        except:
            return False
except:
    client = None
    db = None
    print("Warning: MongoDB connection failed, using mock data")

# Create the main app without a prefix
app = FastAPI(title="Jharkhand Tourism Platform")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize LLM Chat
# Gemini GenAI setup
def get_gemini_response(system_message: str, user_message: str):
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"{system_message}\nUser: {user_message}"
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)

# Pydantic Models
class VendorRegistration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # "hotel", "artisan", "guide"
    location: str
    phone: str
    services: List[str]
    nearby_spots: List[str]
    pricing: Dict[str, Any] = Field(default_factory=dict)
    availability: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Booking(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tourist_name: str
    tourist_phone: str
    vendor_id: str
    vendor_type: str
    service_type: str
    booking_date: str
    message: str = ""
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Feedback(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_type: str  # "tourist", "vendor", "guide"
    vendor_id: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    comment: str
    location: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_message: str
    language: str = "english"  # "english" or "hindi"
    user_type: str = "tourist"  # "tourist", "vendor", "guide", "admin"

class TouristSpot(BaseModel):
    id: str
    name: str
    description: str
    coordinates: Dict[str, float]  # {"lat": float, "lng": float}
    type: str
    city: str
    interest_tags: List[str] = Field(default_factory=list)
    best_time: Optional[str] = None
    duration: Optional[str] = None
    image: Optional[str] = None

class Attraction(BaseModel):
    id: str
    name: str
    city: str
    coordinates: Dict[str, float]
    type: str
    interest_tags: List[str]
    description: str
    best_time: str
    duration: str
    image: str

class Hotel(BaseModel):
    id: str
    name: str
    city: str
    coordinates: Dict[str, float]
    rooms: int
    contact: str
    amenities: List[str]
    price_range: str

# Helper function for sentiment analysis
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Simple sentiment analysis using keywords"""
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "perfect", "beautiful", "nice"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing", "poor", "pathetic"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        score = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = "negative" 
        score = max(0.2, 0.5 - (negative_count - positive_count) * 0.1)
    else:
        sentiment = "neutral"
        score = 0.5
    
    return {"sentiment": sentiment, "score": score}

# Mock data for fallback
MOCK_VENDORS = [
    {
        "id": "vendor1",
        "name": "Ranchi Heritage Hotel",
        "type": "hotel",
        "location": "Ranchi",
        "phone": "+91-9876543210",
        "services": ["Accommodation", "Food", "Travel"],
        "nearby_spots": ["Rock Garden", "Tagore Hill"],
        "pricing": {"per_night": 2500},
        "availability": ["2025-09-22", "2025-09-23"]
    },
    {
        "id": "vendor2",
        "name": "Jharkhand Culture Guide",
        "type": "guide",
        "location": "Ranchi",
        "phone": "+91-8765432109",
        "services": ["Cultural Tours", "Local Sightseeing"],
        "nearby_spots": ["Hundru Falls", "Jonha Falls"],
        "pricing": {"per_day": 1500},
        "availability": ["2025-09-22"]
    }
]

# Routes
@api_router.get("/")
async def root():
    return {"message": "Jharkhand Tourism Platform API", "version": "1.0"}



# Attractions/Tourist Spots endpoints
@api_router.get("/attractions", response_model=List[Attraction])
async def get_attractions(city: Optional[str] = None, interest: Optional[str] = None):
    """Get all attractions with optional filtering"""
    if city:
        attractions = get_attractions_by_city(city)
    elif interest:
        attractions = get_attractions_by_interest(interest)
    else:
        attractions = get_all_attractions()
    
    return [Attraction(**attraction) for attraction in attractions]

@api_router.get("/attractions/{attraction_id}", response_model=Attraction)
async def get_attraction(attraction_id: str):
    """Get specific attraction by ID"""
    attraction = get_attraction_by_id(attraction_id)
    if not attraction:
        raise HTTPException(status_code=404, detail="Attraction not found")
    return Attraction(**attraction)

# Legacy tourist spots endpoint for backward compatibility
@api_router.get("/spots", response_model=List[TouristSpot])
async def get_tourist_spots():
    """Get tourist spots (legacy endpoint)"""
    attractions = get_all_attractions()
    spots = []
    for attraction in attractions:
        spot = TouristSpot(
            id=attraction["id"],
            name=attraction["name"],
            description=attraction["description"],
            coordinates=attraction["coordinates"],
            type=attraction["type"],
            city=attraction["city"],
            interest_tags=attraction.get("interest_tags", []),
            best_time=attraction.get("best_time"),
            duration=attraction.get("duration"),
            image=attraction.get("image")
        )
        spots.append(spot)
    return spots

@api_router.get("/hotels", response_model=List[Hotel])
async def get_hotels(city: Optional[str] = None):
    """Get all hotels or filter by city"""
    if city:
        hotels = get_hotels_by_city(city)
    else:
        hotels = get_all_hotels()
    return [Hotel(**hotel) for hotel in hotels]

@api_router.get("/hotels/{hotel_id}", response_model=Hotel)
async def get_hotel(hotel_id: str):
    """Get specific hotel by ID"""
    hotel = get_hotel_by_id(hotel_id)
    if not hotel:
        raise HTTPException(status_code=404, detail="Hotel not found")
    return Hotel(**hotel)

# Vendor Management
@api_router.post("/vendors", response_model=VendorRegistration)
async def register_vendor(vendor: VendorRegistration):
    vendor_dict = vendor.dict()
    vendor_dict['created_at'] = vendor_dict['created_at'].isoformat()
    if db is not None:
        try:
            await db.vendors.insert_one(vendor_dict)
        except:
            pass  # Continue even if database save fails
    return vendor

@api_router.get("/vendors", response_model=List[VendorRegistration])
async def get_vendors(vendor_type: Optional[str] = None, location: Optional[str] = None):
    if db is None:
        # Use mock data when MongoDB is unavailable
        vendors = MOCK_VENDORS.copy()
        if vendor_type:
            vendors = [v for v in vendors if v['type'] == vendor_type]
        if location:
            vendors = [v for v in vendors if location.lower() in v['location'].lower()]
        
        # Add missing fields for VendorRegistration
        for vendor in vendors:
            if 'created_at' not in vendor:
                vendor['created_at'] = datetime.now(timezone.utc)
        
        return [VendorRegistration(**vendor) for vendor in vendors]
    
    try:
        filter_query = {}
        if vendor_type:
            filter_query['type'] = vendor_type
        if location:
            filter_query['location'] = {"$regex": location, "$options": "i"}
        
        vendors = await db.vendors.find(filter_query).to_list(1000)
        for vendor in vendors:
            if isinstance(vendor.get('created_at'), str):
                vendor['created_at'] = datetime.fromisoformat(vendor['created_at'])
        return [VendorRegistration(**vendor) for vendor in vendors]
    except Exception as e:
        # Fallback to mock data on database error
        vendors = MOCK_VENDORS.copy()
        for vendor in vendors:
            if 'created_at' not in vendor:
                vendor['created_at'] = datetime.now(timezone.utc)
        return [VendorRegistration(**vendor) for vendor in vendors]

@api_router.get("/vendors/{vendor_id}")
async def get_vendor(vendor_id: str):
    if db is not None:
        try:
            vendor = await db.vendors.find_one({"id": vendor_id})
            if vendor:
                if isinstance(vendor.get('created_at'), str):
                    vendor['created_at'] = datetime.fromisoformat(vendor['created_at'])
                return VendorRegistration(**vendor)
        except:
            pass
    
    # Fallback to mock data
    for vendor in MOCK_VENDORS:
        if vendor['id'] == vendor_id:
            if 'created_at' not in vendor:
                vendor['created_at'] = datetime.now(timezone.utc)
            return VendorRegistration(**vendor)
    
    raise HTTPException(status_code=404, detail="Vendor not found")

# Booking Management
@api_router.post("/bookings", response_model=Booking)
async def create_booking(booking: Booking):
    booking_dict = booking.dict()
    booking_dict['created_at'] = booking_dict['created_at'].isoformat()
    if db is not None:
        try:
            await db.bookings.insert_one(booking_dict)
        except:
            pass  # Continue even if database save fails
    return booking

@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(vendor_id: Optional[str] = None, status: Optional[str] = None):
    if db is None:
        return []  # Return empty list when no database
    
    try:
        filter_query = {}
        if vendor_id:
            filter_query['vendor_id'] = vendor_id
        if status:
            filter_query['status'] = status
        
        bookings = await db.bookings.find(filter_query).to_list(1000)
        for booking in bookings:
            if isinstance(booking.get('created_at'), str):
                booking['created_at'] = datetime.fromisoformat(booking['created_at'])
        return [Booking(**booking) for booking in bookings]
    except:
        return []  # Return empty list on error

# Feedback Management
@api_router.post("/feedback", response_model=Feedback)
async def submit_feedback(feedback: Feedback):
    feedback_dict = feedback.dict()
    feedback_dict['created_at'] = feedback_dict['created_at'].isoformat()
    if db is not None:
        try:
            await db.feedback.insert_one(feedback_dict)
        except:
            pass  # Continue even if database save fails
    return feedback

@api_router.get("/feedback", response_model=List[Feedback])
async def get_feedback(vendor_id: Optional[str] = None):
    if db is None:
        return []  # Return empty list when no database
    
    try:
        filter_query = {}
        if vendor_id:
            filter_query['vendor_id'] = vendor_id
        
        feedback_list = await db.feedback.find(filter_query).to_list(1000)
        for feedback in feedback_list:
            if isinstance(feedback.get('created_at'), str):
                feedback['created_at'] = datetime.fromisoformat(feedback['created_at'])
        return [Feedback(**feedback) for feedback in feedback_list]
    except:
        return []  # Return empty list on error

# Contact Information
@api_router.get("/contact/{vendor_id}")
async def get_contact(vendor_id: str):
    if db is not None:
        try:
            vendor = await db.vendors.find_one({"id": vendor_id})
            if vendor:
                return {"phone": vendor.get('phone'), "name": vendor.get('name')}
        except:
            pass
    
    # Fallback to mock data
    for vendor in MOCK_VENDORS:
        if vendor['id'] == vendor_id:
            return {"phone": vendor.get('phone'), "name": vendor.get('name')}
    
    raise HTTPException(status_code=404, detail="Vendor not found")

# Multilingual Chatbot
@api_router.post("/chat")
async def chat_with_bot(message: ChatMessage):
    try:
        language_instruction = ""
        if message.language == "hindi":
            language_instruction = "Please respond in Hindi (Devanagari script). "
        system_message = f"{language_instruction}You are a helpful tourism assistant for Jharkhand state in India. Provide information about tourist spots, local culture, food, festivals, and travel tips. Be friendly and informative."
        response = get_gemini_response(system_message, message.user_message)
        
        # Try to save to database if available
        if db is not None:
            try:
                chat_record = {
                    "id": message.id,
                    "user_message": message.user_message,
                    "bot_response": response,
                    "language": message.language,
                    "user_type": message.user_type,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.chat_history.insert_one(chat_record)
            except:
                pass  # Don't fail the chat if database save fails
        
        return {"response": response, "language": message.language}
    except Exception as e:
        # Provide a helpful fallback response
        fallback_responses = {
            "english": "Hello! I'm having some technical difficulties right now, but I'd be happy to tell you about Jharkhand! It's known for its beautiful waterfalls like Hundru Falls and Jonha Falls, rich tribal culture, and the capital city Ranchi. What specific information would you like to know?",
            "hindi": "नमस्ते! मुझे अभी कुछ तकनीकी समस्या हो रही है, लेकिन मैं झारखंड के बारे में बताने में खुश हूँ! यह अपने सुंदर झरनों जैसे हुंद्रू फॉल्स और जोन्हा फॉल्स, समृद्ध आदिवासी संस्कृति, और राजधानी रांची के लिए प्रसिद्ध है। आप किस बारे में जानना चाहते हैं?"
        }
        fallback_response = fallback_responses.get(message.language, fallback_responses["english"])
        return {"response": fallback_response, "language": message.language, "error": str(e)}

# Emergency endpoints
class EmergencySOSRequest(BaseModel):
    location: Dict[str, float]
    message: str = "Emergency SOS"

@api_router.post("/emergency/sos")
async def emergency_sos(request: EmergencySOSRequest):
    emergency_record = {
        "id": str(uuid.uuid4()),
        "location": request.location,
        "message": request.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active"
    }
    
    if db is not None:
        try:
            await db.emergency_alerts.insert_one(emergency_record)
        except:
            pass  # Continue even if database save fails
    
    # In a real implementation, this would trigger notifications to authorities
    return {"status": "SOS sent", "emergency_id": emergency_record["id"]}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)