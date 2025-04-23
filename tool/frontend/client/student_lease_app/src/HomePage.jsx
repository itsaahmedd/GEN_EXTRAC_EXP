import React from 'react';
import { Link } from 'react-router-dom';
import { Box, Typography, Button } from '@mui/material';
import './HomePage.css';

const HomePage = () => (
  <Box className="home-container">
    <Box className="hero">
      <Typography variant="h3" className="hero-title" gutterBottom>
        Welcome to Student Lease AI Bot
      </Typography>
      <Typography variant="h6" className="hero-subtitle" gutterBottom>
        Manage your leases effortlessly. Upload contracts, view previews, and chat with our AI for insights.
      </Typography>
      <Box className="hero-actions">
        <Button variant="contained" component={Link} to="/contracts" className="hero-btn">
          View Contracts
        </Button>
        <Button variant="contained" component={Link} to="/upload" className="hero-btn">
          Upload Contract
        </Button>
      </Box>
    </Box>
    <Typography variant="body1" className="guidance" align="center">
      Get started by browsing your contracts or uploading a new one.
    </Typography>
  </Box>
);

export default HomePage;
