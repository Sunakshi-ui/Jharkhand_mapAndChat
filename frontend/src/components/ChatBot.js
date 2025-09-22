import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ChatBot = ({ language, userRole }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize with welcome message based on language
    const welcomeMessages = {
      english: "Namaste! I'm your Jharkhand tourism assistant. How can I help you explore the beautiful land of forests? 🌲",
      hindi: "नमस्ते! मैं आपका झारखंड पर्यटन सहायक हूं। मैं आपकी वनों की खूबसूरत भूमि का अन्वेषण करने में कैसे मदद कर सकता हूं? 🌲"
    };

    if (messages.length === 0) {
      setMessages([{
        type: 'bot',
        content: welcomeMessages[language] || welcomeMessages.english,
        timestamp: new Date()
      }]);
    }
  }, [language]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = {
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    
    const messageToSend = inputMessage.trim();
    setInputMessage('');

    try {
      const response = await axios.post(`${API}/chat`, {
        user_message: messageToSend,
        language: language,
        user_type: userRole || 'tourist'
      });

      const botMessage = {
        type: 'bot',
        content: response.data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessages = {
        english: "I'm sorry, I'm having trouble connecting right now. Please try again later.",
        hindi: "खुशफहमी, मुझे अभी जुड़ने में परेशानी हो रही है। कृपया बाद में पुनः प्रयास करें।"
      };

      const errorMessage = {
        type: 'bot',
        content: errorMessages[language] || errorMessages.english,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (timestamp) => {
    return new Intl.DateTimeFormat(language === 'hindi' ? 'hi-IN' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(timestamp);
  };

  const translations = {
    english: {
      title: "Jharkhand Tourism Assistant",
      placeholder: "Ask about tourist spots, culture, food...",
      typing: "Assistant is typing..."
    },
    hindi: {
      title: "झारखंड पर्यटन सहायक",
      placeholder: "पर्यटन स्थलों, संस्कृति, भोजन के बारे में पूछें...",
      typing: "सहायक टाइप कर रहा है..."
    }
  };

  const t = translations[language];

  return (
    <div className="chatbot-container">
      {/* Chat Toggle Button */}
      <button
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="chatbot-toggle"
        aria-label="Toggle chat"
        data-testid="chatbot-toggle"
        type="button"
      >
        {isOpen ? <X size={24} /> : <MessageCircle size={24} />}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="chatbot-window">
          {/* Header */}
          <div className="chatbot-header">
            <div className="flex items-center gap-3">
              <MessageCircle size={20} />
              <div>
                <h3 className="font-semibold">{t.title}</h3>
                <p className="text-xs opacity-90">
                  {language === 'hindi' ? 'ऑनलाइन' : 'Online'}
                </p>
              </div>
            </div>
            <button 
              onClick={() => setIsOpen(false)}
              className="text-white hover:opacity-80"
            >
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="chatbot-messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.type}`}
              >
                <div className="message-content">
                  {message.content}
                </div>
                <div className="message-time text-xs opacity-70 mt-1">
                  {formatTime(message.timestamp)}
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                    <span className="text-xs text-gray-500">{t.typing}</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="chatbot-input-container">
            <input
              type="text"
              className="chatbot-input"
              placeholder={t.placeholder}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
              data-testid="chatbot-message-input"
              id="chatbot-input"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !inputMessage.trim()}
              className="chatbot-send"
              data-testid="chatbot-send-button"
              type="button"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatBot;