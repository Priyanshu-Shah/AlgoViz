import React, { useEffect, useState } from 'react';
import { getModels } from '../api';
import { useNavigate } from 'react-router-dom';

function HomePage() {
  const [models, setModels] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    getModels().then(data => setModels(data));
  }, []);

  return (
    <div>
      <h1>Machine Learning Models</h1>
      <ul>
        {models.map(model => (
          <li key={model.id} onClick={() => navigate(`/${model.id}`)}>
            {model.name}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default HomePage;
