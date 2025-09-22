import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';

// Import components
import LandingPage from './components/LandingPage';
import TouristDashboard from './components/TouristDashboard';
import VendorDashboard from './components/VendorDashboard';
import GuideDashboard from './components/GuideDashboard';
import AdminDashboard from './components/AdminDashboard';
import ChatBot from './components/ChatBot';
import OnboardingPage from './components/OnboardingPage';
import PlannerPage from './components/PlannerPage';
import SharedItineraryPage from './components/SharedItineraryPage';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [userRole, setUserRole] = useState(null);
  const [language, setLanguage] = useState('english');

  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage setUserRole={setUserRole} language={language} setLanguage={setLanguage} />} />
          <Route path="/tourist" element={<TouristDashboard language={language} setLanguage={setLanguage} />} />
          <Route path="/vendor" element={<VendorDashboard language={language} />} />
          <Route path="/guide" element={<GuideDashboard language={language} />} />
          <Route path="/admin" element={<AdminDashboard language={language} />} />
          <Route path="/onboarding" element={<OnboardingPage />} />
            <Route path="/planner" element={<PlannerPage />} />
            <Route path="/itinerary/:id" element={<SharedItineraryPage />} />
        </Routes>
        <ChatBot language={language} userRole={userRole} />
      </BrowserRouter>
    </div>
  );
}

export default App;