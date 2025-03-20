import React, { useEffect, useState } from 'react';
import { getModels } from '../api';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

function HomePage(){
  const [models, setModels] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    getModels().then(data => setModels(data));
  }, []);

  return (
    <div className="homepage">
      <h1 className="homepage-title">Machine Learning Models</h1>
      <div className="model-grid">
        {models.map(model => (
          <div
            key={model.id}
            className="model-card"
            onClick={() => navigate(`/${model.id}`)}
          >
            <img
              src={model.imageUrl}
              alt={model.name}
              className="model-image"
            />
            <p className="model-name">{model.name}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default HomePage;