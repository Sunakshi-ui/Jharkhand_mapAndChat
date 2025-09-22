import React, { useState, useEffect } from 'react';
import { Compass, BookOpen, Award, Users, Calendar, Star, MapPin } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const GuideDashboard = ({ language }) => {
  const [activeTab, setActiveTab] = useState('profile');
  const [guideData, setGuideData] = useState({
    name: '',
    languages: [],
    expertise: [],
    experience: '',
    certifications: []
  });
  const [bookings, setBookings] = useState([]);
  const [loading, setLoading] = useState(false);

  const translations = {
    english: {
      title: "Guide Dashboard",
      subtitle: "Share your knowledge and earn by guiding travelers",
      tabs: {
        profile: "Guide Profile",
        exams: "Language Exams",
        history: "History Study",
        bookings: "My Tours",
        earnings: "Earnings"
      },
      profile: {
        name: "Full Name",
        languages: "Languages Spoken",
        expertise: "Areas of Expertise",
        experience: "Years of Experience",
        certifications: "Certifications"
      },
      exams: {
        hindi: "Hindi Proficiency Test",
        english: "English Proficiency Test",
        tribal: "Tribal Languages Test",
        take: "Take Exam",
        retake: "Retake Exam"
      },
      history: {
        ancient: "Ancient Jharkhand History",
        tribal: "Tribal Heritage & Culture",
        freedom: "Freedom Movement in Jharkhand",
        modern: "Modern Development",
        start: "Start Learning"
      }
    },
    hindi: {
      title: "गाइड डैशबोर्ड",
      subtitle: "अपना ज्ञान साझा करें और यात्रियों को गाइड करके कमाएं",
      tabs: {
        profile: "गाइड प्रोफाइल",
        exams: "भाषा परीक्षा",
        history: "इतिहास अध्ययन",
        bookings: "मेरे टूर",
        earnings: "आमदनी"
      },
      profile: {
        name: "पूरा नाम",
        languages: "बोली जाने वाली भाषाएं",
        expertise: "विशेषज्ञता के क्षेत्र",
        experience: "अनुभव के वर्ष",
        certifications: "प्रमाणपत्र"
      },
      exams: {
        hindi: "हिंदी प्रवीणता परीक्षा",
        english: "अंग्रेजी प्रवीणता परीक्षा",
        tribal: "आदिवासी भाषा परीक्षा",
        take: "परीक्षा दें",
        retake: "पुनः परीक्षा दें"
      },
      history: {
        ancient: "प्राचीन झारखंड का इतिहास",
        tribal: "आदिवासी धरोहर और संस्कृति",
        freedom: "झारखंड में स्वतंत्रता संग्राम",
        modern: "आधुनिक विकास",
        start: "सीखना शुरू करें"
      }
    }
  };

  const t = translations[language];

  useEffect(() => {
    fetchBookings();
  }, []);

  const fetchBookings = async () => {
    try {
      const response = await axios.get(`${API}/bookings?vendor_type=guide`);
      setBookings(response.data);
    } catch (error) {
      console.error('Error fetching guide bookings:', error);
    }
  };

  const registerAsGuide = async () => {
    if (!guideData.name.trim()) {
      alert('Please enter your name first');
      return;
    }

    setLoading(true);
    try {
      const vendorData = {
        name: guideData.name,
        type: 'guide',
        location: 'Jharkhand',
        phone: '+91-9999999999', // In real app, get from form
        services: guideData.expertise,
        nearby_spots: ['Hundru Falls', 'Ranchi Rock Garden', 'Jonha Falls'],
        pricing: { 'full_day': 1500, 'half_day': 800 }
      };

      await axios.post(`${API}/vendors`, vendorData);
      alert('Guide profile registered successfully!');
    } catch (error) {
      console.error('Error registering guide:', error);
      alert('Error registering guide profile');
    } finally {
      setLoading(false);
    }
  };

  const takeLanguageExam = (language) => {
    alert(`Starting ${language} proficiency exam. This feature will be available soon!`);
  };

  const startHistoryStudy = (topic) => {
    alert(`Starting ${topic} study module. Educational content will be loaded!`);
  };

  const renderProfile = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Compass className="card-icon" />
        {t.tabs.profile}
      </h3>
      <div className="space-y-4">
        <div className="form-group">
          <label className="form-label">{t.profile.name}</label>
          <input
            type="text"
            className="form-input"
            value={guideData.name}
            onChange={(e) => setGuideData({...guideData, name: e.target.value})}
            placeholder="Enter your full name"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="form-group">
            <label className="form-label">{t.profile.languages}</label>
            <div className="space-y-2">
              {['Hindi', 'English', 'Mundari', 'Santali', 'Ho'].map(lang => (
                <label key={lang} className="flex items-center">
                  <input
                    type="checkbox"
                    className="mr-2"
                    onChange={(e) => {
                      const languages = e.target.checked
                        ? [...guideData.languages, lang]
                        : guideData.languages.filter(l => l !== lang);
                      setGuideData({...guideData, languages});
                    }}
                  />
                  {lang}
                </label>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">{t.profile.expertise}</label>
            <div className="space-y-2">
              {['Nature Tours', 'Cultural Tours', 'Adventure Sports', 'Photography', 'Tribal Villages'].map(area => (
                <label key={area} className="flex items-center">
                  <input
                    type="checkbox"
                    className="mr-2"
                    onChange={(e) => {
                      const expertise = e.target.checked
                        ? [...guideData.expertise, area]
                        : guideData.expertise.filter(ex => ex !== area);
                      setGuideData({...guideData, expertise});
                    }}
                  />
                  {area}
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">{t.profile.experience}</label>
          <select
            className="form-select"
            value={guideData.experience}
            onChange={(e) => setGuideData({...guideData, experience: e.target.value})}
          >
            <option value="">Select experience level</option>
            <option value="0-1">0-1 years</option>
            <option value="2-5">2-5 years</option>
            <option value="5+">5+ years</option>
          </select>
        </div>

        <button 
          onClick={registerAsGuide}
          disabled={loading}
          className="btn w-full"
        >
          {loading ? 'Registering...' : 'Register as Guide'}
        </button>
      </div>
    </div>
  );

  const renderExams = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Award className="card-icon" />
        {t.tabs.exams}
      </h3>
      <div className="space-y-4">
        <p className="text-gray-600">Take language proficiency exams to improve your guide credentials</p>
        
        <div className="grid gap-4">
          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.exams.hindi}</h4>
              <p className="text-sm text-gray-600">Test your Hindi speaking and comprehension skills</p>
            </div>
            <button 
              onClick={() => takeLanguageExam('Hindi')}
              className="btn"
            >
              {t.exams.take}
            </button>
          </div>

          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.exams.english}</h4>
              <p className="text-sm text-gray-600">Assess your English communication abilities</p>
            </div>
            <button 
              onClick={() => takeLanguageExam('English')}
              className="btn"
            >
              {t.exams.take}
            </button>
          </div>

          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.exams.tribal}</h4>
              <p className="text-sm text-gray-600">Demonstrate knowledge of local tribal languages</p>
            </div>
            <button 
              onClick={() => takeLanguageExam('Tribal Languages')}
              className="btn"
            >
              {t.exams.take}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderHistory = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <BookOpen className="card-icon" />
        {t.tabs.history}
      </h3>
      <div className="space-y-4">
        <p className="text-gray-600">Study Jharkhand's rich history to become a better guide</p>
        
        <div className="grid gap-4">
          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.history.ancient}</h4>
              <p className="text-sm text-gray-600">Learn about early settlements and kingdoms</p>
            </div>
            <button 
              onClick={() => startHistoryStudy('Ancient History')}
              className="btn"
            >
              {t.history.start}
            </button>
          </div>

          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.history.tribal}</h4>
              <p className="text-sm text-gray-600">Understand tribal traditions and customs</p>
            </div>
            <button 
              onClick={() => startHistoryStudy('Tribal Heritage')}
              className="btn"
            >
              {t.history.start}
            </button>
          </div>

          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.history.freedom}</h4>
              <p className="text-sm text-gray-600">Discover Jharkhand's role in independence</p>
            </div>
            <button 
              onClick={() => startHistoryStudy('Freedom Movement')}
              className="btn"
            >
              {t.history.start}
            </button>
          </div>

          <div className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
            <div>
              <h4 className="font-semibold">{t.history.modern}</h4>
              <p className="text-sm text-gray-600">Learn about recent developments and progress</p>
            </div>
            <button 
              onClick={() => startHistoryStudy('Modern Development')}
              className="btn"
            >
              {t.history.start}
            </button>
          </div>
        </div>
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
          <div className="text-center py-8">
            <Users className="mx-auto text-gray-400 mb-4" size={48} />
            <p className="text-gray-500">No tour bookings yet</p>
            <p className="text-sm text-gray-400">Register as a guide to start receiving bookings</p>
          </div>
        ) : (
          bookings.map(booking => (
            <div key={booking.id} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="font-semibold text-gray-900">{booking.tourist_name}</h4>
                  <div className="text-sm text-gray-600 mt-1">
                    <p className="flex items-center gap-1">
                      <MapPin size={14} />
                      {booking.service_type}
                    </p>
                    <p>Date: {booking.booking_date}</p>
                    <p>Phone: {booking.tourist_phone}</p>
                    {booking.message && <p>Request: {booking.message}</p>}
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
                  <button className="btn text-sm">Accept</button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderEarnings = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Star className="card-icon" />
        {t.tabs.earnings}
      </h3>
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h4 className="font-semibold text-green-800">This Month</h4>
            <p className="text-2xl font-bold text-green-600">₹0</p>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-800">Total Tours</h4>
            <p className="text-2xl font-bold text-blue-600">0</p>
          </div>
          <div className="bg-yellow-50 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-800">Rating</h4>
            <p className="text-2xl font-bold text-yellow-600">-</p>
          </div>
        </div>
        
        <div className="text-center py-8">
          <p className="text-gray-500">Start guiding tours to see your earnings here!</p>
        </div>
      </div>
    </div>
  );

  const tabComponents = {
    profile: renderProfile,
    exams: renderExams,
    history: renderHistory,
    bookings: renderBookings,
    earnings: renderEarnings
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

export default GuideDashboard;