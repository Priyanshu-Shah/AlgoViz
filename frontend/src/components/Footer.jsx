import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

function Footer() {
  const year = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          <p className="copyright">© {year} AlgoViz</p>
          <p className="tagline">Exploring machine learning algorithms through interactive visualization.</p>
        </div>
        <div className="footer-links">
          <Link to="/about" className="footer-link">About</Link>
          <Link to="/docs" className="footer-link">Documentation</Link>
          <a 
            href="https://github.com/Priyanshu-Shah/AlgoViz" 
            target="_blank" 
            rel="noopener noreferrer"
            className="footer-link"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
