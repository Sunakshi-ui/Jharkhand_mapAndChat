import React, { useState, useEffect } from 'react';
import { Store, MapPin, Calendar, Star, Phone, Users, DollarSign } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const VendorDashboard = ({ language }) => {
  const [activeTab, setActiveTab] = useState('listings');
  const [vendors, setVendors] = useState([]);
  const [bookings, setBookings] = useState([]);
  const [feedback, setFeedback] = useState([]);
  const [newVendor, setNewVendor] = useState({
    name: '',
    type: 'hotel',
    location: '',
    phone: '',
    services: [],
    nearby_spots: [],
    pricing: {}
  });
  const [loading, setLoading] = useState(false);

  const translations = {
    english: {
      title: "Vendor & Hotel Dashboard",
      subtitle: "Manage your listings and connect with travelers",
      tabs: {
        listings: "My Listings",
        bookings: "Bookings",
        feedback: "Feedback",
        add: "Add Listing"
      },
      form: {
        name: "Business Name",
        type: "Business Type",
        location: "Location",
        phone: "Phone Number",
        services: "Services Offered",
        nearbySpots: "Nearby Tourist Spots",
        pricing: "Pricing Information"
      },
      types: {
        hotel: "Hotel",
        artisan: "Artisan Shop",
        restaurant: "Restaurant",
        transport: "Transport Service"
      }
    },
    hindi: {
      title: "विक्रेता और होटल डैशबोर्ड",
      subtitle: "अपनी सूची प्रबंधित करें और यात्रियों से जुड़ें",
      tabs: {
        listings: "मेरी सूचियां",
        bookings: "बुकिंग",
        feedback: "फीडबैक",
        add: "सूची जोड़ें"
      },
      form: {
        name: "व्यापार का नाम",
        type: "व्यापार का प्रकार",
        location: "स्थान",
        phone: "फोन नंबर",
        services: "प्रदान की जाने वाली सेवाएं",
        nearbySpots: "नजदीकी पर्यटन स्थल",
        pricing: "मूल्य निर्धारण की जानकारी"
      },
      types: {
        hotel: "होटल",
        artisan: "कारीगर की दुकान",
        restaurant: "रेस्टोरेंट",
        transport: "परिवहन सेवा"
      }
    }
  };

  const t = translations[language];

  useEffect(() => {
    fetchVendors();
    fetchBookings();
    fetchFeedback();
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

  const fetchFeedback = async () => {
    try {
      const response = await axios.get(`${API}/feedback`);
      setFeedback(response.data);
    } catch (error) {
      console.error('Error fetching feedback:', error);
    }
  };

  const handleAddVendor = async (e) => {
    e.preventDefault();
    if (!newVendor.name.trim() || !newVendor.location.trim()) return;

    setLoading(true);
    try {
      await axios.post(`${API}/vendors`, newVendor);
      alert('Listing added successfully!');
      setNewVendor({
        name: '',
        type: 'hotel',
        location: '',
        phone: '',
        services: [],
        nearby_spots: [],
        pricing: {}
      });
      fetchVendors();
      setActiveTab('listings');
    } catch (error) {
      console.error('Error adding vendor:', error);
      alert('Error adding listing');
    } finally {
      setLoading(false);
    }
  };

  const renderListings = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Store className="card-icon" />
        {t.tabs.listings}
      </h3>
      <div className="space-y-4">
        {vendors.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No listings yet. Add your first listing!</p>
        ) : (
          vendors.slice(0, 5).map(vendor => (
            <div key={vendor.id} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900">{vendor.name}</h4>
                  <div className="flex items-center gap-4 mt-1 text-sm text-gray-600">
                    <span className="flex items-center gap-1">
                      <Store size={14} />
                      {t.types[vendor.type] || vendor.type}
                    </span>
                    <span className="flex items-center gap-1">
                      <MapPin size={14} />
                      {vendor.location}
                    </span>
                    <span className="flex items-center gap-1">
                      <Phone size={14} />
                      {vendor.phone}
                    </span>
                  </div>
                  <div className="mt-2">
                    <p className="text-sm text-blue-600">
                      Services: {vendor.services.join(', ')}
                    </p>
                    {vendor.nearby_spots.length > 0 && (
                      <p className="text-sm text-green-600">
                        Near: {vendor.nearby_spots.join(', ')}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button className="btn btn-secondary text-sm">Edit</button>
                  <button className="btn text-sm">View Stats</button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderBookings = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Calendar className="card-icon" />
        {t.tabs.bookings}
      </h3>
      <div className="space-y-4">
        {bookings.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No bookings yet.</p>
        ) : (
          bookings.slice(0, 5).map(booking => (
            <div key={booking.id} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="font-semibold text-gray-900">{booking.tourist_name}</h4>
                  <div className="text-sm text-gray-600 mt-1">
                    <p>Service: {booking.service_type}</p>
                    <p>Date: {booking.booking_date}</p>
                    <p>Phone: {booking.tourist_phone}</p>
                    {booking.message && <p>Message: {booking.message}</p>}
                  </div>
                </div>
                <div className="flex gap-2">
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    booking.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                    booking.status === 'confirmed' ? 'bg-green-100 text-green-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {booking.status}
                  </span>
                  <button className="btn text-sm">Contact</button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderFeedback = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Star className="card-icon" />
        {t.tabs.feedback}
      </h3>
      <div className="space-y-4">
        {feedback.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No feedback yet.</p>
        ) : (
          feedback.slice(0, 5).map(fb => (
            <div key={fb.id} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="flex">
                      {[1, 2, 3, 4, 5].map(star => (
                        <Star
                          key={star}
                          size={16}
                          className={star <= fb.rating ? 'text-yellow-500 fill-current' : 'text-gray-300'}
                        />
                      ))}
                    </div>
                    <span className="text-sm text-gray-600">({fb.rating}/5)</span>
                  </div>
                  <p className="text-gray-700">{fb.comment}</p>
                  <p className="text-xs text-gray-500 mt-2">
                    {fb.user_type} • {fb.location}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderAddListing = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Users className="card-icon" />
        {t.tabs.add}
      </h3>
      <form onSubmit={handleAddVendor} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="form-group">
            <label className="form-label">{t.form.name}</label>
            <input
              type="text"
              className="form-input"
              value={newVendor.name}
              onChange={(e) => setNewVendor({...newVendor, name: e.target.value})}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">{t.form.type}</label>
            <select
              className="form-select"
              value={newVendor.type}
              onChange={(e) => setNewVendor({...newVendor, type: e.target.value})}
            >
              <option value="hotel">Hotel</option>
              <option value="artisan">Artisan Shop</option>
              <option value="restaurant">Restaurant</option>
              <option value="transport">Transport</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="form-group">
            <label className="form-label">{t.form.location}</label>
            <input
              type="text"
              className="form-input"
              value={newVendor.location}
              onChange={(e) => setNewVendor({...newVendor, location: e.target.value})}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">{t.form.phone}</label>
            <input
              type="tel"
              className="form-input"
              value={newVendor.phone}
              onChange={(e) => setNewVendor({...newVendor, phone: e.target.value})}
            />
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">{t.form.services}</label>
          <input
            type="text"
            className="form-input"
            placeholder="Enter services separated by commas"
            onChange={(e) => setNewVendor({
              ...newVendor, 
              services: e.target.value.split(',').map(s => s.trim()).filter(s => s)
            })}
          />
        </div>

        <div className="form-group">
          <label className="form-label">{t.form.nearbySpots}</label>
          <input
            type="text"
            className="form-input"
            placeholder="Enter nearby tourist spots separated by commas"
            onChange={(e) => setNewVendor({
              ...newVendor, 
              nearby_spots: e.target.value.split(',').map(s => s.trim()).filter(s => s)
            })}
          />
        </div>

        <button 
          type="submit"
          disabled={loading}
          className="btn w-full"
        >
          {loading ? 'Adding...' : 'Add Listing'}
        </button>
      </form>
    </div>
  );

  const tabComponents = {
    listings: renderListings,
    bookings: renderBookings,
    feedback: renderFeedback,
    add: renderAddListing
  };

  return (
    <div className="dashboard-container">
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

export default VendorDashboard;