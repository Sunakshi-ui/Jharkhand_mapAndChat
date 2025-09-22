import React from 'react';
import { useNavigate } from 'react-router-dom';
import { MapPin, Users, Compass, Shield, Camera, Heart } from 'lucide-react';

const LandingPage = ({ setUserRole, language, setLanguage }) => {
  const navigate = useNavigate();

  const handleRoleSelection = (role) => {
    setUserRole(role);
    navigate(`/${role}`);
  };

  const translations = {
    english: {
      title: "Discover Jharkhand",
      subtitle: "Your Cultural Compass & Storyteller for the Land of Forests",
      tagline: "Not just a travel app. A cultural compass. A storyteller. A practical companion.",
      roles: {
        tourist: {
          title: "Use as Tourist",
          description: "Explore Jharkhand with interactive maps, itinerary planning, and immersive cultural content"
        },
        vendor: {
          title: "Use as Hotel Vendor", 
          description: "Manage your hotel listings, bookings, and connect with travelers visiting Jharkhand"
        },
        artisan: {
          title: "Use as Local Artisan",
          description: "Showcase your crafts, connect with tourists, and preserve traditional Jharkhand culture"
        },
        guide: {
          title: "Use as Guide",
          description: "Offer guided tours, share local knowledge, and earn by helping travelers explore"
        },
        admin: {
          title: "Use as Government Admin",
          description: "Monitor tourism trends, analyze feedback, and make data-driven policy decisions"
        }
      }
    },
    hindi: {
      title: "झारखंड की खोज करें",
      subtitle: "वनों की भूमि के लिए आपका सांस्कृतिक कंपास और कहानीकार",
      tagline: "सिर्फ एक ट्रैवल ऐप नहीं। एक सांस्कृतिक कंपास। एक कहानीकार। एक व्यावहारिक साथी।",
      roles: {
        tourist: {
          title: "पर्यटक के रूप में उपयोग करें",
          description: "इंटरैक्टिव मैप्स, यात्रा योजना और सांस्कृतिक सामग्री के साथ झारखंड का अन्वेषण करें"
        },
        vendor: {
          title: "होटल विक्रेता के रूप में उपयोग करें",
          description: "अपने होटल की सूची, बुकिंग प्रबंधित करें और झारखंड आने वाले यात्रियों से जुड़ें"
        },
        artisan: {
          title: "स्थानीय कारीगर के रूप में उपयोग करें",
          description: "अपने शिल्प का प्रदर्शन करें, पर्यटकों से जुड़ें और पारंपरिक झारखंडी संस्कृति को संरक्षित करें"
        },
        guide: {
          title: "गाइड के रूप में उपयोग करें",
          description: "गाइडेड टूर ऑफर करें, स्थानीय ज्ञान साझा करें और यात्रियों की मदद करके कमाएं"
        },
        admin: {
          title: "सरकारी व्यवस्थापक के रूप में उपयोग करें",
          description: "पर्यटन रुझानों की निगरानी करें, फीडबैक का विश्लेषण करें और डेटा-संचालित नीति निर्णय लें"
        }
      }
    }
  };

  const t = translations[language];

  const roleIcons = {
    tourist: <Camera size={32} />,
    vendor: <MapPin size={32} />,
    artisan: <Heart size={32} />,
    guide: <Compass size={32} />,
    admin: <Shield size={32} />
  };

  return (
    <div className="landing-container">
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

      <div className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title font-display">{t.title}</h1>
          <p className="hero-subtitle font-inter">{t.subtitle}</p>
          <p className="text-lg font-medium text-blue-800 mb-8">{t.tagline}</p>

          <div className="role-grid">
            {Object.entries(t.roles).map(([roleKey, role]) => (
              <div 
                key={roleKey}
                className="role-card"
                onClick={() => handleRoleSelection(roleKey === 'artisan' ? 'vendor' : roleKey)}
              >
                <div className="role-icon">
                  {roleIcons[roleKey]}
                </div>
                <h3 className="role-title font-display">{role.title}</h3>
                <p className="role-description font-inter">{role.description}</p>
              </div>
            ))}
          </div>

          {/* Cultural Elements */}
          <div className="mt-16 text-center">
            <div className="flex flex-wrap justify-center gap-8 items-center opacity-70">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <MapPin size={16} />
                <span>Waterfalls • Tribal Culture</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Users size={16} />
                <span>Local Artisans • Folk Music</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Heart size={16} />
                <span>Handloom • Traditional Crafts</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;