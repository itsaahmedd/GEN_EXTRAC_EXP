import React from 'react';
import { NavLink } from 'react-router-dom';
import { Box, Typography } from '@mui/material';
import './NavBar.css';

const NavBar = () => {
  return (
    <Box className="navbar">
      <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
        Lease AI Bot
      </Typography>
      <Box className="nav-links">
        <NavLink to="/" className="nav-link">
          Home
        </NavLink>
        <NavLink to="/contracts" className="nav-link">
          Contracts
        </NavLink>
        <NavLink to="/upload" className="nav-link">
          Upload
        </NavLink>
      </Box>
    </Box>
  );
};

export default NavBar;
