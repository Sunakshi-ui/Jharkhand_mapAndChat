import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { Calendar, Map, Users, Camera, AlertTriangle, Phone, Star, BookOpen } from 'lucide-react';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import OnboardingPage from './OnboardingPage';
import placeholderIconUrl from '../assets/placeholder.png';
const customIcon = L.icon({
  iconUrl: placeholderIconUrl,
  iconSize: [35, 35],
  iconAnchor: [17, 35],
  popupAnchor: [0, -35],
});

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TouristDashboard = ({ language, setLanguage }) => {
  const [activeTab, setActiveTab] = useState('itinerary');
  const [vendors, setVendors] = useState([]);
  const [bookings, setBookings] = useState([]);
  const [feedback, setFeedback] = useState({ rating: 5, comment: '' });
  const [loading, setLoading] = useState(false);
  const [attractions, setAttractions] = useState([]);
  const [onboardingData, setOnboardingData] = useState(null);
  const [selectedAttractions, setSelectedAttractions] = useState([]);

  const translations = {
    english: {
      title: "Tourist Dashboard",
      subtitle: "Plan, Explore & Experience Jharkhand",
      tabs: {
        itinerary: "Itinerary Planner",
        attractions: "Attractions",
        map: "Interactive Map",
        bookings: "Bookings",
        content: "Cultural Content",
        emergency: "Emergency"
      },
      buttons: {
        bookNow: "Book Now",
        callGuide: "Call Guide", 
        submitFeedback: "Submit Feedback",
        emergency: "SOS",
        shareLocation: "Share Live Location"
      },
      placeholders: {
        feedback: "Share your experience about Jharkhand...",
        destination: "Where do you want to go?",
        dates: "Select travel dates"
      }
    },
    hindi: {
      title: "पर्यटक डैशबोर्ड",
      subtitle: "योजना बनाएं, अन्वेषण करें और झारखंड का अनुभव करें",
      tabs: {
        itinerary: "यात्रा योजनाकार",
        attractions: "आकर्षण",
        map: "इंटरैक्टिव मैप",
        bookings: "बुकिंग",
        content: "सांस्कृतिक सामग्री",
        emergency: "आपातकाल"
      },
      buttons: {
        bookNow: "अभी बुक करें",
        callGuide: "गाइड को कॉल करें",
        submitFeedback: "फीडबैक जमा करें",
        emergency: "SOS",
        shareLocation: "लाइव लोकेशन साझा करें"
      },
      placeholders: {
        feedback: "झारखंड के बारे में अपना अनुभव साझा करें...",
        destination: "आप कहाँ जाना चाहते हैं?",
        dates: "यात्रा की तारीख चुनें"
      }
    }
  };

  const t = translations[language];

  useEffect(() => {
    fetchVendors();
    fetchBookings();
    fetchAttractions();
  }, []);

  const fetchVendors = async () => {
    try {
      const response = await axios.get(`${API}/vendors`);
      setVendors(response.data);
    } catch (error) {
      console.error('Error fetching vendors:', error);
    }
  };

  const fetchBookings = async () => {
    try {
      const response = await axios.get(`${API}/bookings`);
      setBookings(response.data);
    } catch (error) {
      console.error('Error fetching bookings:', error);
    }
  };

  const fetchAttractions = async () => {
    try {
      const response = await axios.get(`${API}/attractions`);
      setAttractions(response.data);
    } catch (error) {
      console.error('Error fetching attractions:', error);
      // Fallback to hardcoded data if API fails
      setAttractions([
        { id: 'fallback1', name: "Hundru Falls", coordinates: { lat: 23.4241, lng: 85.6186 } },
        { id: 'fallback2', name: "Ranchi Rock Garden", coordinates: { lat: 23.3441, lng: 85.3096 } },
        { id: 'fallback3', name: "Jonha Falls", coordinates: { lat: 23.3176, lng: 85.4629 } },
        { id: 'fallback4', name: "Patratu Valley", coordinates: { lat: 23.6693, lng: 84.8369 } }
      ]);
    }
  };

  const handleBooking = async (vendor) => {
    setLoading(true);
    try {
      const bookingData = {
        tourist_name: "Tourist User", // In real app, get from auth
        tourist_phone: "+91-9999999999",
        vendor_id: vendor.id,
        vendor_type: vendor.type,
        service_type: vendor.services[0] || "general",
        booking_date: new Date().toISOString().split('T')[0],
        message: "Booking request from tourist dashboard"
      };

      await axios.post(`${API}/bookings`, bookingData);
      alert('Booking request sent successfully!');
      fetchBookings();
    } catch (error) {
      console.error('Error creating booking:', error);
      alert('Error creating booking');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (!feedback.comment.trim()) return;
    
    setLoading(true);
    try {
      await axios.post(`${API}/feedback`, {
        user_type: "tourist",
        rating: feedback.rating,
        comment: feedback.comment,
        location: "Jharkhand"
      });
      
      alert('Feedback submitted successfully!');
      setFeedback({ rating: 5, comment: '' });
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Error submitting feedback');
    } finally {
      setLoading(false);
    }
  };

  const handleEmergency = async () => {
    try {
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject);
      });
      
      const location = {
        lat: position.coords.latitude,
        lng: position.coords.longitude
      };
      
      await axios.post(`${API}/emergency/sos`, {
        location: location,
        message: "Emergency SOS from tourist"
      });
      
      alert('SOS sent! Help is on the way.');
    } catch (error) {
      console.error('Error sending SOS:', error);
      alert('Error sending emergency alert');
    }
  };

  const getContactInfo = async (vendorId) => {
    try {
      const response = await axios.get(`${API}/contact/${vendorId}`);
      alert(`Contact: ${response.data.name}\nPhone: ${response.data.phone}`);
    } catch (error) {
      console.error('Error getting contact info:', error);
    }
  };

  // Map spots are now loaded from API via attractions state
const renderMap = () => { return(
    <div className="dashboard-card">
      <h3 className="card-title">
        <Map className="card-icon" />
        {t.tabs.map}
      </h3>
      <div className="map-container">
        <MapContainer 
          center={[24.481445, 86.697462]} 
          zoom={7} 
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {attractions.map(attraction => (
            <Marker 
              key={attraction.id} 
              position={[attraction.coordinates.lat, attraction.coordinates.lng]}
              icon={customIcon}
            >
              <Popup>
                <div>
                  <h4 className="font-semibold">{attraction.name}</h4>
                  <p className="text-sm text-gray-600">{attraction.city} • {attraction.type}</p>
                  <p className="text-xs">{attraction.description}</p>
                  {attraction.best_time && (
                    <p className="text-xs text-blue-600">Best time: {attraction.best_time}</p>
                  )}
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
      <div className="mt-4">
        <button className="btn btn-secondary mr-2">Enable Offline Mode</button>
        <button className="btn">Get Directions</button>
      </div>
    </div>)}
  const [showOnboarding, setShowOnboarding] = useState(false);
<button className="btn" onClick={() => setShowOnboarding(true)}>
  Launch Onboarding
</button>

  const renderItineraryPlanner = () => {
  if (showOnboarding) {
    return <OnboardingPage />;
  }

  return (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Calendar className="card-icon" />
        {t.tabs.itinerary}
      </h3>
      <button className="btn" onClick={() => setShowOnboarding(true)}>
        Launch Onboarding
      </button>
      {/* You can keep or remove the rest of the itinerary form here */}
    </div>
  );
};
<button onClick={() => setShowOnboarding(false)}>Back to Dashboard</button>


  const renderBookings = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Users className="card-icon" />
        Available Services
      </h3>
      <div className="space-y-4">
        {vendors.slice(0, 3).map(vendor => (
          <div key={vendor.id} className="p-4 border border-gray-200 rounded-lg">
            <div className="flex justify-between items-start">
              <div>
                <h4 className="font-semibold text-gray-900">{vendor.name}</h4>
                <p className="text-sm text-gray-600">{vendor.type} • {vendor.location}</p>
                <p className="text-sm text-blue-600">{vendor.services.join(', ')}</p>
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={() => handleBooking(vendor)}
                  disabled={loading}
                  className="btn text-sm"
                >
                  {loading ? 'Booking...' : t.buttons.bookNow}
                </button>
                <button 
                  onClick={() => getContactInfo(vendor.id)}
                  className="btn btn-secondary text-sm"
                >
                  <Phone size={16} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderAttractions = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Map className="card-icon" />
        Popular Attractions
      </h3>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {attractions.slice(0, 6).map(attraction => (
          <div key={attraction.id} className="p-3 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h4 className="font-semibold text-gray-900">{attraction.name}</h4>
                <p className="text-sm text-gray-600">{attraction.city} • {attraction.type}</p>
                <p className="text-sm text-gray-700 mt-1">{attraction.description}</p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {attraction.interest_tags && attraction.interest_tags.map(tag => (
                    <span key={tag} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                      {tag}
                    </span>
                  ))}
                </div>
                {attraction.best_time && (
                  <p className="text-xs text-green-600 mt-1">Best time: {attraction.best_time}</p>
                )}
              </div>
              <div className="ml-3">
                <button className="btn btn-secondary text-sm">View Details</button>
              </div>
            </div>
          </div>
        ))}
        {attractions.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <Map className="mx-auto mb-2 text-gray-400" size={32} />
            <p>Loading attractions...</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderCulturalContent = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Camera className="card-icon" />
        Cultural Content
      </h3>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gradient-to-r from-orange-50 to-red-50 p-4 rounded-lg">
            <BookOpen className="text-orange-600 mb-2" />
            <h4 className="font-semibold">Tribal Heritage</h4>
            <p className="text-sm text-gray-600">Learn about indigenous communities</p>
          </div>
          <div className="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg">
            <Camera className="text-green-600 mb-2" />
            <h4 className="font-semibold">360° Previews</h4>
            <p className="text-sm text-gray-600">Virtual tours of landmarks</p>
          </div>
        </div>
        
        <div className="form-group">
          <label className="form-label">Rate Your Experience</label>
          <div className="flex gap-1 mb-2">
            {[1, 2, 3, 4, 5].map(star => (
              <button
                key={star}
                onClick={() => setFeedback({...feedback, rating: star})}
                className={`p-1 ${star <= feedback.rating ? 'text-yellow-500' : 'text-gray-300'}`}
              >
                <Star size={20} fill={star <= feedback.rating ? 'currentColor' : 'none'} />
              </button>
            ))}
          </div>
          <textarea
            className="form-input form-textarea"
            placeholder={t.placeholders.feedback}
            value={feedback.comment}
            onChange={(e) => setFeedback({...feedback, comment: e.target.value})}
          />
          <button 
            onClick={submitFeedback}
            disabled={loading}
            className="btn mt-2"
          >
            {loading ? 'Submitting...' : t.buttons.submitFeedback}
          </button>
        </div>
      </div>
    </div>
  );

  const renderEmergency = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <AlertTriangle className="card-icon text-red-600" />
        {t.tabs.emergency}
      </h3>
      <div className="space-y-4 text-center">
        <p className="text-gray-600">Emergency assistance and safety features</p>
        <div className="grid grid-cols-1 gap-4">
          <button 
            onClick={handleEmergency}
            className="btn bg-red-600 hover:bg-red-700 text-white py-4"
          >
            <AlertTriangle size={20} />
            {t.buttons.emergency}
          </button>
          <button className="btn btn-warning">
            {t.buttons.shareLocation}
          </button>
          <div className="text-sm text-gray-500">
            <p>Emergency Contacts:</p>
            <p>Police: 100 | Fire: 101 | Ambulance: 102</p>
          </div>
        </div>
      </div>
    </div>
  );

  const tabComponents = {
    itinerary: renderItineraryPlanner,
    attractions: renderAttractions,
    map: renderMap,
    bookings: renderBookings,
    content: renderCulturalContent,
    emergency: renderEmergency
  };

  return (
    <div className="dashboard-container">
      {/* Language Toggle */}
      <div className="language-toggle">
        <button 
          className={`language-btn ${language === 'english' ? 'active' : ''}`}
          onClick={() => setLanguage('english')}
        >
          English
        </button>
        <button 
          className={`language-btn ${language === 'hindi' ? 'active' : ''}`}
          onClick={() => setLanguage('hindi')}
        >
          हिंदी
        </button>
      </div>

      <div className="dashboard-header">
        <h1 className="dashboard-title font-display">{t.title}</h1>
        <p className="dashboard-subtitle font-inter">{t.subtitle}</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center mb-8">
        <div className="bg-white rounded-lg p-1 shadow-lg">
          {Object.entries(t.tabs).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === key 
                  ? 'bg-blue-600 text-white shadow' 
                  : 'text-gray-600 hover:text-blue-600'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Active Tab Content */}
      <div className="max-w-4xl mx-auto">
        {tabComponents[activeTab]()}
      </div>
    </div>
  );
};

export default TouristDashboard;
