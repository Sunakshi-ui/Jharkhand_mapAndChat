import React from 'react';
import { useParams } from 'react-router-dom';

const SharedItineraryPage = () => {
  const { id } = useParams();

  return (
    <div className="shared-itinerary-page">
      <div className="container">
        <div className="text-center">
          <h1>Shared Itinerary</h1>
          <p>Itinerary ID: {id}</p>
          <p>🚧 This page will display shared itineraries from other users</p>
        </div>
      </div>
    </div>
  );
};

export default SharedItineraryPage;