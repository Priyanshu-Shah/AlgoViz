import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import LinReg from './pages/LinReg';
import KNN from './pages/KNN';
import DicTree from './pages/DecTree';
import About from './pages/About';
import Docs from './pages/Docs';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lin-reg" element={<LinReg />} />
          <Route path="/knn" element={<KNN />} />
          <Route path="/dec-tree" element={<DicTree />} />
          <Route path="/about" element={<About />} />
          <Route path="/docs" element={<Docs />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
