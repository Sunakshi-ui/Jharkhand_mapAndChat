import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Users, MapPin, MessageSquare, Shield, AlertCircle } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AdminDashboard = ({ language }) => {
  const [activeTab, setActiveTab] = useState('analytics');
  const [analytics, setAnalytics] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [loading, setLoading] = useState(false);

  const translations = {
    english: {
      title: "Government Admin Dashboard",
      subtitle: "Tourism insights, policy tools, and regional analytics",
      tabs: {
        analytics: "Analytics",
        sentiment: "Sentiment Analysis",
        trends: "Booking Trends",
        regions: "Regional Data",
        policies: "Policy Tools"
      },
      metrics: {
        totalBookings: "Total Bookings",
        totalVendors: "Registered Vendors",
        avgRating: "Average Rating",
        totalFeedback: "Total Feedback"
      }
    },
    hindi: {
      title: "सरकारी व्यवस्थापक डैशबोर्ड",
      subtitle: "पर्यटन अंतर्दृष्टि, नीति उपकरण और क्षेत्रीय विश्लेषण",
      tabs: {
        analytics: "विश्लेषण",
        sentiment: "भावना विश्लेषण",
        trends: "बुकिंग रुझान",
        regions: "क्षेत्रीय डेटा",
        policies: "नीति उपकरण"
      },
      metrics: {
        totalBookings: "कुल बुकिंग",
        totalVendors: "पंजीकृत विक्रेता",
        avgRating: "औसत रेटिंग",
        totalFeedback: "कुल फीडबैक"
      }
    }
  };

  const t = translations[language];

  useEffect(() => {
    fetchAnalytics();
    fetchSentimentAnalysis();
  }, []);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/analytics`);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSentimentAnalysis = async () => {
    try {
      const response = await axios.get(`${API}/sentiment`);
      setSentimentData(response.data);
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
    }
  };

  const renderAnalytics = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="dashboard-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t.metrics.totalBookings}</p>
              <p className="text-2xl font-bold text-blue-600">
                {analytics?.total_bookings || 0}
              </p>
            </div>
            <BarChart3 className="text-blue-500" size={32} />
          </div>
        </div>

        <div className="dashboard-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t.metrics.totalVendors}</p>
              <p className="text-2xl font-bold text-green-600">
                {analytics?.total_vendors || 0}
              </p>
            </div>
            <Users className="text-green-500" size={32} />
          </div>
        </div>

        <div className="dashboard-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t.metrics.avgRating}</p>
              <p className="text-2xl font-bold text-yellow-600">
                {sentimentData?.average_rating?.toFixed(1) || 'N/A'}
              </p>
            </div>
            <TrendingUp className="text-yellow-500" size={32} />
          </div>
        </div>

        <div className="dashboard-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t.metrics.totalFeedback}</p>
              <p className="text-2xl font-bold text-purple-600">
                {sentimentData?.total_feedback || 0}
              </p>
            </div>
            <MessageSquare className="text-purple-500" size={32} />
          </div>
        </div>
      </div>

      {/* Booking Breakdown */}
      <div className="dashboard-card">
        <h3 className="card-title">
          <BarChart3 className="card-icon" />
          Booking Distribution by Type
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {analytics?.booking_by_type && Object.entries(analytics.booking_by_type).map(([type, count]) => (
            <div key={type} className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 capitalize">{type}</p>
              <p className="text-xl font-bold text-blue-600">{count}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Vendor Distribution */}
      <div className="dashboard-card">
        <h3 className="card-title">
          <Users className="card-icon" />
          Vendor Distribution by Type
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {analytics?.vendors_by_type && Object.entries(analytics.vendors_by_type).map(([type, count]) => (
            <div key={type} className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 capitalize">{type}</p>
              <p className="text-xl font-bold text-green-600">{count}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderSentimentAnalysis = () => (
    <div className="space-y-6">
      <div className="dashboard-card">
        <h3 className="card-title">
          <MessageSquare className="card-icon" />
          {t.tabs.sentiment}
        </h3>
        
        {sentimentData ? (
          <div className="space-y-6">
            {/* Sentiment Chart */}
            {sentimentData.chart_image && (
              <div className="text-center">
                <img 
                  src={sentimentData.chart_image} 
                  alt="Sentiment Analysis Chart"
                  className="max-w-full h-auto mx-auto rounded-lg"
                />
              </div>
            )}

            {/* Sentiment Distribution */}
            <div className="grid grid-cols-3 gap-4">
              {sentimentData.sentiment_distribution && Object.entries(sentimentData.sentiment_distribution).map(([sentiment, count]) => (
                <div key={sentiment} className={`text-center p-4 rounded-lg ${
                  sentiment === 'positive' ? 'bg-green-50' :
                  sentiment === 'negative' ? 'bg-red-50' :
                  'bg-gray-50'
                }`}>
                  <p className={`text-sm font-medium ${
                    sentiment === 'positive' ? 'text-green-800' :
                    sentiment === 'negative' ? 'text-red-800' :
                    'text-gray-800'
                  }`}>
                    {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                  </p>
                  <p className={`text-2xl font-bold ${
                    sentiment === 'positive' ? 'text-green-600' :
                    sentiment === 'negative' ? 'text-red-600' :
                    'text-gray-600'
                  }`}>
                    {count}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <MessageSquare className="mx-auto text-gray-400 mb-4" size={48} />
            <p className="text-gray-500">No feedback data available for sentiment analysis</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderTrends = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <TrendingUp className="card-icon" />
        {t.tabs.trends}
      </h3>
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-blue-800">Peak Season</h4>
            <p className="text-blue-600">October - March</p>
            <p className="text-sm text-blue-600">75% of annual bookings</p>
          </div>
          <div className="p-4 bg-green-50 rounded-lg">
            <h4 className="font-semibold text-green-800">Popular Activities</h4>
            <p className="text-green-600">Waterfall Tours, Tribal Villages</p>
            <p className="text-sm text-green-600">60% preference rate</p>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="font-semibold">Monthly Trends</h4>
          <div className="h-40 bg-gray-100 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Trend chart will be displayed here</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderRegionalData = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <MapPin className="card-icon" />
        {t.tabs.regions}
      </h3>
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { region: 'Ranchi District', bookings: 45, vendors: 23, rating: 4.2 },
            { region: 'East Singhbhum', bookings: 32, vendors: 18, rating: 4.0 },
            { region: 'West Singhbhum', bookings: 28, vendors: 15, rating: 4.1 },
            { region: 'Dhanbad', bookings: 20, vendors: 12, rating: 3.9 }
          ].map(region => (
            <div key={region.region} className="p-4 border border-gray-200 rounded-lg">
              <h4 className="font-semibold text-gray-900">{region.region}</h4>
              <div className="mt-2 space-y-1 text-sm text-gray-600">
                <p>Bookings: {region.bookings}</p>
                <p>Vendors: {region.vendors}</p>
                <p>Avg Rating: {region.rating}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderPolicyTools = () => (
    <div className="dashboard-card">
      <h3 className="card-title">
        <Shield className="card-icon" />
        {t.tabs.policies}
      </h3>
      <div className="space-y-4">
        <div className="grid gap-4">
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertCircle className="text-yellow-600 mt-0.5" size={20} />
              <div>
                <h4 className="font-semibold text-yellow-800">Content Moderation</h4>
                <p className="text-sm text-yellow-700">Review and moderate user-generated content</p>
                <button className="btn btn-warning mt-2 text-sm">Review Content</button>
              </div>
            </div>
          </div>

          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start gap-3">
              <Users className="text-blue-600 mt-0.5" size={20} />
              <div>
                <h4 className="font-semibold text-blue-800">Vendor Verification</h4>
                <p className="text-sm text-blue-700">Verify and approve new vendor registrations</p>
                <button className="btn text-sm mt-2">Verify Vendors</button>
              </div>
            </div>
          </div>

          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-start gap-3">
              <BarChart3 className="text-green-600 mt-0.5" size={20} />
              <div>
                <h4 className="font-semibold text-green-800">Tourism Reports</h4>
                <p className="text-sm text-green-700">Generate detailed tourism reports for policy making</p>
                <button className="btn btn-success text-sm mt-2">Generate Report</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const tabComponents = {
    analytics: renderAnalytics,
    sentiment: renderSentimentAnalysis,
    trends: renderTrends,
    regions: renderRegionalData,
    policies: renderPolicyTools
  };

  if (loading && !analytics) {
    return (
      <div className="dashboard-container">
        <div className="flex justify-center items-center h-64">
          <div className="loading text-blue-600">Loading analytics...</div>
        </div>
      </div>
    );
  }

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
      <div className="max-w-6xl mx-auto">
        {tabComponents[activeTab]()}
      </div>
    </div>
  );
};

export default AdminDashboard;