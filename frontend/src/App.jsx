import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import LinReg from './pages/LinReg';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/lin-reg" element={<LinReg />} />
      </Routes>
    </Router>
  );
}

export default App;
