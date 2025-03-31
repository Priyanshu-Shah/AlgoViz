import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import LinReg from './pages/LinReg';
import KNN from './pages/KNN';
import DicTree from './pages/DecTree';
import KMeans from './pages/KMeans';
import About from './pages/About';
import Docs from './pages/Docs';
import PCA from './pages/PCA';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lin-reg" element={<LinReg />} />
          <Route path="/knn" element={<KNN />} />
          <Route path="/dec-tree" element={<DicTree />} />
          <Route path="/kmeans" element={<KMeans />} />
          <Route path="/about" element={<About />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/pca" element={<PCA />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
