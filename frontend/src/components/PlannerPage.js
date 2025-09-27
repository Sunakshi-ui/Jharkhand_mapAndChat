
import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import axios from 'axios';
import './PlannerPage.css';

// Leaflet and React-Leaflet Imports
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';

// Import the single placeholder icon
import placeholderIconUrl from '../assets/placeholder.png';
import hotel from '../assets/hotel.png';

// Backend API configuration
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Custom icon setup for all markers
const customIcon = L.icon({
  iconUrl: placeholderIconUrl,
  iconSize: [35, 35],
  iconAnchor: [17, 35],
  popupAnchor: [0, -35],
});
const hotelIcon = L.icon({
  iconUrl: hotel,
  iconSize: [35, 35],
  iconAnchor: [17, 35],
  popupAnchor: [0, -35],
});

const PlannerPage = () => {
  const location = useLocation();
  const [onboardingData, setOnboardingData] = useState(null);
  const [filteredAttractions, setFilteredAttractions] = useState([]);
  const [allAttractions, setAllAttractions] = useState([]);
  const [itinerary, setItinerary] = useState({});
  const [routeData, setRouteData] = useState({});
  const [selectedInterestFilter, setSelectedInterestFilter] = useState('all');

  useEffect(() => {
    if (location.state?.onboardingData) {
      setOnboardingData(location.state.onboardingData);
      const destination = location.state.onboardingData.destination;

      // Calculate the number of days from the start and end dates
      const startDate = new Date(location.state.onboardingData.startDate);
      const endDate = new Date(location.state.onboardingData.endDate);
      const diffTime = Math.abs(endDate - startDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1; // +1 to include the start day

      const initialItinerary = {};
      for (let i = 1; i <= diffDays; i++) {
        initialItinerary[`Day ${i}`] = [];
      }
      setItinerary(initialItinerary);

      axios.get(`${API}/attractions?city=${destination}`)
        .then(response => {
          setAllAttractions(response.data);
          setFilteredAttractions(response.data);
        })
        .catch(error => {
          console.error("Error fetching attractions:", error);
          // Fallback data if API fails
          const fallbackData = [
            {
              id: 'fallback1',
              name: 'Hundru Falls',
              coordinates: { lat: 23.4230, lng: 85.5979 },
              city: destination,
              interest_tags: ['Adventure', 'Relaxation'],
              description: 'A spectacular waterfall with a drop of 98 meters',
              type: 'waterfall'
            }
          ];
          setAllAttractions(fallbackData);
          setFilteredAttractions(fallbackData);
        });
    }
  }, [location]);

  useEffect(() => {
    // This effect runs whenever the itinerary changes
    const fetchRoutes = async () => {
      const newRouteData = {};
      const promises = Object.keys(itinerary).map(async (day) => {
        const locations = itinerary[day];
        if (locations.length > 1) {
          try {
            // For this example, we'll simulate route data.
            newRouteData[day] = locations.map(loc => [loc.coordinates.lat, loc.coordinates.lng]);
          } catch (error) {
            console.error(`Error fetching route for ${day}:`, error);
          }
        }
      });
      await Promise.all(promises);
      setRouteData(newRouteData);
    };

    fetchRoutes();
  }, [itinerary]);

  // Filter attractions by interest
  const filterByInterest = (interest) => {
    setSelectedInterestFilter(interest);
    if (interest === 'all') {
      setFilteredAttractions(allAttractions);
    } else {
      const filtered = allAttractions.filter(attraction => 
        attraction.interest_tags && attraction.interest_tags.includes(interest)
      );
      setFilteredAttractions(filtered);
    }
  };

  const handleDragEnd = result => {
    const { source, destination } = result;
    if (!destination) return;

    const sourceDay = source.droppableId;
    const destDay = destination.droppableId;
    const draggedItem = itinerary[sourceDay][source.index];

    const newItinerary = { ...itinerary };
    newItinerary[sourceDay] = Array.from(newItinerary[sourceDay]);
    newItinerary[sourceDay].splice(source.index, 1);

    newItinerary[destDay] = Array.from(newItinerary[destDay]);
    newItinerary[destDay].splice(destination.index, 0, draggedItem);

    setItinerary(newItinerary);
  };

  const addToDay = (attraction, day) => {
    // Check if attraction is already in the day
    if (itinerary[day].some(item => item.id === attraction.id)) {
      alert(`${attraction.name} is already added to ${day}!`);
      return;
    }
    
    // Check if attraction is already in another day
    const existingDay = Object.keys(itinerary).find(d => 
      itinerary[d].some(item => item.id === attraction.id)
    );
    
    if (existingDay) {
      const confirmMove = window.confirm(
        `${attraction.name} is already in ${existingDay}. Do you want to move it to ${day}?`
      );
      if (confirmMove) {
        // Remove from existing day and add to new day
        setItinerary(prev => ({
          ...prev,
          [existingDay]: prev[existingDay].filter(item => item.id !== attraction.id),
          [day]: [...prev[day], attraction],
        }));
      }
    } else {
      // Add to the selected day
      setItinerary(prev => ({
        ...prev,
        [day]: [...prev[day], attraction],
      }));
    }
  };

  const [hotels, setHotels] = useState([]);

useEffect(() => {
  fetch(`${API}/hotels`)
    .then(res => res.json())
    .then(data => setHotels(data));
}, []);


const dayColors = ["blue", "green", "red", "orange", "purple", "teal"];
  return (
    <div className="planner-page">
      <div className="container">
        <div className="planner-header">
          <h1>Your Jharkhand Itinerary</h1>
          <p>Drag and drop attractions to plan your perfect trip</p>
        </div>
        {onboardingData && (
          <div className="onboarding-summary card">
            <div className="card-body">
              <h3>Your Preferences</h3>
              <p><strong>Dates:</strong> {new Date(onboardingData.startDate).toDateString()} - {new Date(onboardingData.endDate).toDateString()}</p>
              <p><strong>Route:</strong> {onboardingData.currentLocation} → {onboardingData.destination}</p>
            </div>
          </div>
        )}

        <div className="planner-content">
          <div className="attractions-panel">
            <h3>Available Attractions in {onboardingData?.destination}</h3>
            
            {/* Interest Filter */}
            <div className="interest-filters">
              <h4>Filter by Interest:</h4>
              <div className="filter-buttons">
                <button 
                  className={`filter-btn ${selectedInterestFilter === 'all' ? 'active' : ''}`}
                  onClick={() => filterByInterest('all')}
                >
                  All ({allAttractions.length})
                </button>
                {onboardingData?.interests?.map(interest => {
                  const count = allAttractions.filter(a => 
                    a.interest_tags && a.interest_tags.includes(interest)
                  ).length;
                  return (
                    <button 
                      key={interest}
                      className={`filter-btn ${selectedInterestFilter === interest ? 'active' : ''}`}
                      onClick={() => filterByInterest(interest)}
                    >
                      {interest} ({count})
                    </button>
                  );
                })}
              </div>
            </div>
            
            <div className="attractions-list">
              {filteredAttractions.map((attraction) => (
                <div key={attraction.id} className="attraction-card card">
                  <div className="card-body">
                    <h4>{attraction.name}</h4>
                    <p className="attraction-description">{attraction.description}</p>
                    <div className="attraction-tags">
                      {attraction.interest_tags && attraction.interest_tags.map(tag => (
                        <span key={tag} className="tag">{tag}</span>
                      ))}
                    </div>
                    <p className="attraction-location"><em>{attraction.city} • {attraction.type}</em></p>
                    {attraction.best_time && (
                      <p className="best-time">Best time: {attraction.best_time}</p>
                    )}
                    <div className="day-buttons">
                      {Object.keys(itinerary).map(day => (
                        <button 
                          key={day} 
                          className="btn btn-sm btn-outline"
                          onClick={() => addToDay(attraction, day)}
                          disabled={itinerary[day].some(item => item.id === attraction.id)}
                        >
                          {itinerary[day].some(item => item.id === attraction.id) ? '✓' : '+'} {day}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
              {filteredAttractions.length === 0 && (
                <div className="no-attractions">
                  <p>Loading attractions for {onboardingData?.destination}...</p>
                </div>
              )}
            </div>
          </div>

          <div className="itinerary-panel">
            <h3>Your Itinerary</h3>
            <DragDropContext onDragEnd={handleDragEnd}>
              {Object.keys(itinerary).map(day => (
                <Droppable droppableId={day} key={day}>
                  {(provided) => (
                    <div className="day-card card" ref={provided.innerRef} {...provided.droppableProps}>
                      <div className="card-header">
                        <h4>{day}</h4>
                      </div>
                      <div className="card-body">
                        {itinerary[day].map((item, index) => (
                          <Draggable key={item.id} draggableId={item.id} index={index}>
                            {(provided) => (
                              <div
                                className="draggable-item"
                                ref={provided.innerRef}
                                {...provided.draggableProps}
                                {...provided.dragHandleProps}
                              >
                                {item.name}
                              </div>
                            )}
                          </Draggable>
                        ))}
                        {provided.placeholder}
                      </div>
                    </div>
                  )}
                </Droppable>
              ))}
            </DragDropContext>
          </div>

          <div className="map-panel">
            <h3>Map View</h3>
            <div className='leaflet-container'>
              <MapContainer center={[24.481445, 86.697462]} zoom={8} style={{ height: "600px", width: "100%" }}>


                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Markers for all filtered attractions */}
                {filteredAttractions.map(attraction => (
                  <Marker
                    key={attraction.id}
                    position={[attraction.coordinates.lat, attraction.coordinates.lng]}
                    icon={customIcon}
                  >
                    <Popup>
                      <div>
                        <h4>{attraction.name}</h4>
                        <p>{attraction.description}</p>
                        <p><strong>Tags:</strong> {attraction.interest_tags.join(', ')}</p>
                      </div>
                    </Popup>
                  </Marker>
                ))}

                {hotels.map(hotel => (
  <Marker
    key={hotel.id}
    position={[hotel.coordinates.lat, hotel.coordinates.lng]}
    icon={hotelIcon} // define a distinct icon for hotels
  >
    <Popup>
      <div>
        <h4>{hotel.name}</h4>
        <p><strong>City:</strong> {hotel.city}</p>
        <p><strong>Rooms:</strong> {hotel.rooms}</p>
        <p><strong>Contact:</strong> {hotel.contact}</p>
        <p><strong>Price:</strong> {hotel.price_range}</p>
        <p><strong>Amenities:</strong> {hotel.amenities.join(', ')}</p>
      </div>
    </Popup>
  </Marker>
))}

                {/* Draw polylines for each day's itinerary */}
                {Object.keys(itinerary).map((day, index) => {
  const locations = itinerary[day];
  if (locations.length > 1) {
    const positions = locations.map(loc => [loc.coordinates.lat, loc.coordinates.lng]);
    return (
      <Polyline
        key={day}
        positions={positions}
        color={dayColors[index % dayColors.length]}
        weight={5}
      />
    );
  }
  return null;
})}

              </MapContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlannerPage;