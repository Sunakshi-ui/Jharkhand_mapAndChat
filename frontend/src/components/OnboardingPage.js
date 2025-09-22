import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import './OnboardingPage.css';

const OnboardingPage = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    interests: [],
    startDate: null,
    endDate: null,
    currentLocation: '',
    destination: ''
  });

  const interests = [
    { id: 'Adventure', label: 'Adventure', icon: '🏔️', description: 'Trekking, waterfalls, outdoor activities' },
    { id: 'Culture', label: 'Culture', icon: '🏛️', description: 'Museums, historical sites, local traditions' },
    { id: 'Food', label: 'Food', icon: '🍽️', description: 'Local cuisine, street food, restaurants' },
    { id: 'Spirituality', label: 'Spirituality', icon: '🕉️', description: 'Temples, meditation, sacred places' },
    { id: 'Relaxation', label: 'Relaxation', icon: '🧘', description: 'Parks, lakes, peaceful environments' }
  ];

  const cities = ['Ranchi', 'Jamshedpur', 'Hazaribagh', 'Sahibganj', 'Dhanbad'];

  const handleInterestToggle = (interestId) => {
    setFormData(prev => ({
      ...prev,
      interests: prev.interests.includes(interestId)
        ? prev.interests.filter(id => id !== interestId)
        : [...prev.interests, interestId]
    }));
  };

  const handleNext = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    } else {
      // Navigate to planner with form data
      navigate('/planner', { state: { onboardingData: formData } });
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const isStepValid = () => {
    switch (currentStep) {
      case 1:
        return formData.interests.length > 0;
      case 2:
        return formData.startDate && formData.endDate;
      case 3:
        return formData.currentLocation && formData.destination;
      default:
        return false;
    }
  };

  return (
    <div className="onboarding-page">
      <div className="container">
        <div className="onboarding-header">
          <h1>Plan Your Jharkhand Adventure</h1>
          <div className="progress-bar">
            <div className="progress-steps">
              {[1, 2, 3].map(step => (
                <div key={step} className={`step ${currentStep >= step ? 'active' : ''}`}>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="onboarding-content">
          {/* Step 1: Interest Selection */}
          {currentStep === 1 && (
            <div className="step-content">
              <h2>What interests you most?</h2>
              <p>Select all that apply to get personalized recommendations</p>
              
              <div className="interests-grid">
                {interests.map(interest => (
                  <div
                    key={interest.id}
                    className={`interest-card ${formData.interests.includes(interest.id) ? 'selected' : ''}`}
                    onClick={() => handleInterestToggle(interest.id)}
                  >
                    <div className="interest-icon">{interest.icon}</div>
                    <h3>{interest.label}</h3>
                    <p>{interest.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Step 2: Date Selection */}
          {currentStep === 2 && (
            <div className="step-content">
              <h2>When are you planning to visit?</h2>
              <p>Choose your travel dates to get the best recommendations</p>
              
              <div className="date-selection">
                <div className="form-group">
                  <label className="form-label">Start Date</label>
                  <DatePicker
                    selected={formData.startDate}
                    onChange={(date) => setFormData(prev => ({ ...prev, startDate: date }))}
                    className="form-control"
                    placeholderText="Select start date"
                    minDate={new Date()}
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">End Date</label>
                  <DatePicker
                    selected={formData.endDate}
                    onChange={(date) => setFormData(prev => ({ ...prev, endDate: date }))}
                    className="form-control"
                    placeholderText="Select end date"
                    minDate={formData.startDate || new Date()}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Location Selection */}
          {currentStep === 3 && (
            <div className="step-content">
              <h2>Where are you traveling from and to?</h2>
              <p>This helps us suggest attractions along your route</p>
              
              <div className="location-selection">
                <div className="form-group">
                  <label className="form-label">Current Location</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Enter your starting location"
                    value={formData.currentLocation}
                    onChange={(e) => setFormData(prev => ({ ...prev, currentLocation: e.target.value }))}
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Primary Destination in Jharkhand</label>
                  <select
                    className="form-control"
                    value={formData.destination}
                    onChange={(e) => setFormData(prev => ({ ...prev, destination: e.target.value }))}
                  >
                    <option value="">Select a destination</option>
                    {cities.map(city => (
                      <option key={city} value={city}>{city}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="onboarding-actions">
          {currentStep > 1 && (
            <button className="btn btn-secondary" onClick={handleBack}>
              Back
            </button>
          )}
          
          <button
            className="btn btn-primary"
            onClick={handleNext}
            disabled={!isStepValid()}
          >
            {currentStep === 3 ? 'Create My Itinerary' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default OnboardingPage;