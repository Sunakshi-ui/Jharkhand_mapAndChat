
import google.generativeai as genai
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
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
matplotlib.use('Agg') # Use non-interactive backend
import seaborn as sns

# --- App Initialization and Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Initialize FastAPI app and router
app = FastAPI(title="Jharkhand Tourism Platform")
api_router = APIRouter(prefix="/api")

# Add CORS middleware with a specific origin for better security
# This allows requests from your React development server on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001"
]
,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM Chat with API Key from environment
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
try:
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logger.error(f"Failed to initialize LLM model: {e}")
    llm_model = None # Fallback to a helpful message if model fails to load

# --- In-Memory Data Store ---
# Using Python dictionaries and lists to simulate a database.
# This data will be reset every time the server restarts.
db = {
    "vendors": [
        {
            "id": "vendor1",
            "name": "Ranchi Heritage Hotel",
            "type": "hotel",
            "location": "Ranchi",
            "phone": "+91-9876543210",
            "services": ["Accommodation", "Food", "Travel"],
            "nearby_spots": ["Rock Garden", "Tagore Hill"],
            "pricing": {"per_night": 2500},
            "availability": ["2025-09-22", "2025-09-23"],
            "created_at": datetime.now(timezone.utc).isoformat()
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
            "availability": ["2025-09-22"],
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "vendor3",
            "name": "Adventure Trekkers",
            "type": "tourist",
            "location": "Betla National Park",
            "phone": "+91-9988776655",
            "services": ["Jungle Safari", "Trekking"],
            "nearby_spots": ["Betla Fort"],
            "pricing": {"per_person": 1200},
            "availability": ["2025-09-24"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    ],
    "bookings": [
        {
            "id": "booking1",
            "tourist_name": "Alice",
            "tourist_phone": "1234567890",
            "vendor_id": "vendor1",
            "vendor_type": "hotel",
            "service_type": "Accommodation",
            "booking_date": "2025-09-22",
            "message": "Double room, please.",
            "status": "confirmed",
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "booking2",
            "tourist_name": "Bob",
            "tourist_phone": "0987654321",
            "vendor_id": "vendor2",
            "vendor_type": "guide",
            "service_type": "Local Sightseeing",
            "booking_date": "2025-09-23",
            "message": "Looking for a day tour.",
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    ],
    "feedback": [
        {
            "id": "feedback1",
            "user_type": "tourist",
            "vendor_id": "vendor1",
            "rating": 5,
            "comment": "Had a wonderful stay, the staff was very friendly and helpful. Great experience!",
            "location": "Ranchi",
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "feedback2",
            "user_type": "tourist",
            "vendor_id": "vendor2",
            "rating": 4,
            "comment": "The guide was knowledgeable, a great tour. Really enjoyed it.",
            "location": "Ranchi",
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "feedback3",
            "user_type": "tourist",
            "vendor_id": "vendor1",
            "rating": 2,
            "comment": "The room was a bit noisy and the service was slow. It was a bad experience.",
            "location": "Ranchi",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    ],
    "tourist_spots": [
        {
            "id": "spot1",
            "name": "Hundru Falls",
            "description": "One of the most scenic waterfalls in Jharkhand.",
            "location": {"lat": 23.4475, "lng": 85.5034},
            "category": "nature",
            "images": ["hundru_falls_1.jpg"],
            "facilities": ["parking", "food stalls"]
        },
        {
            "id": "spot2",
            "name": "Betla National Park",
            "description": "A national park known for its rich wildlife.",
            "location": {"lat": 23.8967, "lng": 84.1867},
            "category": "adventure",
            "images": ["betla_national_park_1.jpg"],
            "facilities": ["safari rides", "restrooms"]
        }
    ],
    "emergency_alerts": [],
    "chat_history": []
}


# --- Pydantic Models ---
class VendorRegistration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # "hotel", "artisan", "guide", "tourist"
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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    location: Dict[str, float]  # {"lat": float, "lng": float}
    category: str  # "nature", "culture", "adventure", "religious"
    images: List[str] = Field(default_factory=list)
    facilities: List[str] = Field(default_factory=list)

class EmergencySOSRequest(BaseModel):
    location: Dict[str, float]
    message: str = "Emergency SOS"

# --- Helper Functions ---
def get_gemini_response(system_message: str, user_message: str):
    """Generates a response using the globally-initialized Gemini model."""
    if not llm_model:
        raise RuntimeError("LLM model is not initialized.")
    prompt = f"{system_message}\nUser: {user_message}"
    response = llm_model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)

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

# --- API Routes ---
@api_router.get("/")
async def root():
    return {"message": "Jharkhand Tourism Platform API", "version": "1.0"}

# Vendor Management
@api_router.post("/vendors", response_model=VendorRegistration)
async def register_vendor(vendor: VendorRegistration):
    vendor_dict = vendor.dict()
    vendor_dict['created_at'] = datetime.now(timezone.utc).isoformat()
    db['vendors'].append(vendor_dict)
    return VendorRegistration(**vendor_dict)

@api_router.get("/vendors", response_model=List[VendorRegistration])
async def get_vendors(vendor_type: Optional[str] = None, location: Optional[str] = None):
    vendors = db['vendors']
    if vendor_type:
        vendors = [v for v in vendors if v['type'] == vendor_type]
    if location:
        vendors = [v for v in vendors if location.lower() in v['location'].lower()]
    return [VendorRegistration(**v) for v in vendors]

@api_router.get("/vendors/{vendor_id}", response_model=VendorRegistration)
async def get_vendor(vendor_id: str):
    vendor = next((v for v in db['vendors'] if v['id'] == vendor_id), None)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return VendorRegistration(**vendor)

# Booking Management
@api_router.post("/bookings", response_model=Booking)
async def create_booking(booking: Booking):
    booking_dict = booking.dict()
    booking_dict['created_at'] = datetime.now(timezone.utc).isoformat()
    db['bookings'].append(booking_dict)
    return Booking(**booking_dict)

@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(vendor_id: Optional[str] = None, status: Optional[str] = None):
    bookings = db['bookings']
    if vendor_id:
        bookings = [b for b in bookings if b['vendor_id'] == vendor_id]
    if status:
        bookings = [b for b in bookings if b['status'] == status]
    return [Booking(**b) for b in bookings]

# Feedback Management
@api_router.post("/feedback", response_model=Feedback)
async def submit_feedback(feedback: Feedback):
    feedback_dict = feedback.dict()
    feedback_dict['created_at'] = datetime.now(timezone.utc).isoformat()
    db['feedback'].append(feedback_dict)
    return Feedback(**feedback_dict)

@api_router.get("/feedback", response_model=List[Feedback])
async def get_feedback(vendor_id: Optional[str] = None):
    feedback_list = db['feedback']
    if vendor_id:
        feedback_list = [f for f in feedback_list if f['vendor_id'] == vendor_id]
    return [Feedback(**f) for f in feedback_list]

# Sentiment Analysis
@api_router.get("/sentiment")
async def get_sentiment_analysis():
    feedback_list = db['feedback']
    
    if not feedback_list:
        return {"message": "No feedback data available"}
    
    sentiments = []
    ratings = []
    
    for feedback in feedback_list:
        sentiment_result = analyze_sentiment(feedback['comment'])
        sentiments.append(sentiment_result)
        ratings.append(feedback['rating'])
    
    # Create sentiment distribution chart
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment['sentiment']] += 1
    
    # Generate chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sentiment pie chart
    ax1.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    ax1.set_title('Feedback Sentiment Distribution')
    
    # Rating histogram
    sns.histplot(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ax=ax2, color='skyblue')
    ax2.set_xlabel('Rating')
    ax2.set_ylabel('Count')
    ax2.set_title('Rating Distribution')
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "sentiment_distribution": sentiment_counts,
        "average_rating": sum(ratings) / len(ratings) if ratings else 0,
        "total_feedback": len(feedback_list),
        "chart_image": f"data:image/png;base64,{image_base64}"
    }

# Analytics for Admin
@api_router.get("/analytics")
async def get_analytics():
    # Get booking trends
    bookings = db['bookings']
    vendors = db['vendors']
    
    booking_counts_by_type = {}
    vendor_counts_by_type = {}
    
    for booking in bookings:
        vendor_type = booking.get('vendor_type', 'unknown')
        booking_counts_by_type[vendor_type] = booking_counts_by_type.get(vendor_type, 0) + 1
    
    for vendor in vendors:
        vendor_type = vendor.get('type', 'unknown')
        vendor_counts_by_type[vendor_type] = vendor_counts_by_type.get(vendor_type, 0) + 1
    
    return {
        "total_bookings": len(bookings),
        "total_vendors": len(vendors),
        "booking_by_type": booking_counts_by_type,
        "vendors_by_type": vendor_counts_by_type
    }

# Contact Information
@api_router.get("/contact/{vendor_id}")
async def get_contact(vendor_id: str):
    vendor = next((v for v in db['vendors'] if v['id'] == vendor_id), None)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return {"phone": vendor.get('phone'), "name": vendor.get('name')}

# Multilingual Chatbot
@api_router.post("/chat")
async def chat_with_bot(message: ChatMessage):
    fallback_responses = {
        "english": "Hello! I'm having some technical difficulties right now, but I'd be happy to tell you about Jharkhand! It's known for its beautiful waterfalls like Hundru Falls and Jonha Falls, rich tribal culture, and the capital city Ranchi. What specific information would you like to know?",
        "hindi": "नमस्ते! मुझे अभी कुछ तकनीकी समस्या हो रही है, लेकिन मैं झारखंड के बारे में बताने में खुश हूँ! यह अपने सुंदर झरनों जैसे हुंद्रू फॉल्स और जोन्हा फॉल्स, समृद्ध आदिवासी संस्कृति, और राजधानी रांची के लिए प्रसिद्ध है। आप किस बारे में जानना चाहते हैं?"
    }

    try:
        language_instruction = "Please respond in Hindi (Devanagari script). " if message.language == "hindi" else ""
        system_message = f"{language_instruction}You are a helpful tourism assistant for Jharkhand state in India. Provide information about tourist spots, local culture, food, festivals, and travel tips. Be friendly and informative."
        response = get_gemini_response(system_message, message.user_message)
        
        # Save chat history to in-memory store
        chat_record = {
            "id": message.id,
            "user_message": message.user_message,
            "bot_response": response,
            "language": message.language,
            "user_type": message.user_type,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        db['chat_history'].append(chat_record)
        
        return {"response": response, "language": message.language}
    except Exception as e:
        logger.error(f"LLM API call failed, providing fallback response. Error: {e}")
        fallback_response = fallback_responses.get(message.language, fallback_responses["english"])
        return {"response": fallback_response, "language": message.language, "error": str(e)}

# Tourist Spots
@api_router.get("/spots", response_model=List[TouristSpot])
async def get_tourist_spots():
    return [TouristSpot(**spot) for spot in db['tourist_spots']]

@api_router.post("/spots", response_model=TouristSpot)
async def add_tourist_spot(spot: TouristSpot):
    spot_dict = spot.dict()
    db['tourist_spots'].append(spot_dict)
    return TouristSpot(**spot_dict)

# Emergency endpoints
@api_router.post("/emergency/sos")
async def emergency_sos(request: EmergencySOSRequest):
    emergency_record = {
        "id": str(uuid.uuid4()),
        "location": request.location,
        "message": request.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active"
    }
    db['emergency_alerts'].append(emergency_record)
    
    return {"status": "SOS sent", "emergency_id": emergency_record["id"]}

# Include the router in the main app
app.include_router(api_router)

# Example command to run the application:
# uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
