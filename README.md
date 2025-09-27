# Jharkhand Tourism Platform

A full-stack web application for exploring tourist attractions, planning itineraries, chatting with a tourism assistant, and managing bookings in Jharkhand, India.

## Features
- Interactive map and tourist spot listings
- AI-powered chatbot for travel queries
- Vendor, guide, and tourist dashboards
- Booking and feedback management
- Admin analytics and emergency SOS

## Technologies Used
- **Frontend:** React, Tailwind CSS
- **Backend:** FastAPI, Google Generative AI (Gemini), Python
- **Database:** In-memory (demo) or MongoDB (production)

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js & npm

### Backend Setup
1. Navigate to the backend folder:
   ```powershell
   cd backend
   ```
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env` (see `.env.example` if available).
5. Run the backend server:
   ```powershell
   python server_integrated.py
   ```
   or
   ```powershell
   uvicorn server_integrated:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend folder:
   ```powershell
   cd frontend
   ```
2. Install dependencies:
   ```powershell
   npm install
   ```
3. Start the frontend:
   ```powershell
   npm start
   ```

### Accessing the App
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000/api](http://localhost:8000/api)

## Environment Variables
- `GEMINI_API_KEY`: Google Generative AI API key

## Project Structure
```
Jharkhand_mapAndChat-main/
├── backend/
│   ├── server_integrated.py
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
