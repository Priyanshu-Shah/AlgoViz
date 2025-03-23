import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import LinReg from './pages/LinReg';
import DicTree  from './pages/DecTree';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/lin-reg" element={<LinReg />} />
        <Route path="/dec-tree" element={<DicTree />} />
      </Routes>
    </Router>
  );
}

export default App;
