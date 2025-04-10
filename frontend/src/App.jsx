import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import LinReg from './pages/LinReg';
import KNN from './pages/KNN';
import DTrees from './pages/DTrees';
import KMeans from './pages/KMeans';
import About from './pages/About';
import Docs from './pages/Docs';
import PCA from './pages/PCA';
import SVM from './pages/SVM';
import DBScan from './pages/DBScan';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lin-reg" element={<LinReg />} />
          <Route path="/knn" element={<KNN />} />
          <Route path="/d-trees" element={<DTrees />} />
          <Route path="/kmeans" element={<KMeans />} />
          <Route path="/about" element={<About />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/pca" element={<PCA />} />
          <Route path='/svm' element={<SVM />} />
          <Route path='/DBScan' element={<DBScan />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;

