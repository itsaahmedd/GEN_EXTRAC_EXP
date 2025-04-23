import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import NavBar from './NavBar';
import HomePage from './HomePage';
import ListContracts from './ListContracts';
import PDFUpload from './PDFUpload';
import ChatbotLayout from './ChatbotLayout';
import './App.css';
import theme from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <div className="app-container">
          <header className="header">
            <NavBar />
          </header>
          <main className="main-content">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/contracts" element={<ListContracts />} />
              <Route path="/upload" element={<PDFUpload />} />
              <Route path="/chatbot/:id" element={<ChatbotLayout />} />
            </Routes>
          </main>
          <footer className="footer">
            <img
              src="/logo.png"
              alt="Logo"
              style={{
                position: 'absolute',
                left: '16px',
                height: 90,
                width: 215,
              }}
            />
            <p style={{ textAlign: 'center', width: '100%' }}>
              &copy; {new Date().getFullYear()} Student Lease AI Bot. All rights reserved.
            </p>
          </footer>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
