import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { Box, Button, Typography, Paper, CircularProgress } from '@mui/material';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { toast } from 'react-toastify';
import './PDFUpload.css';


import config from './config';
const apiEndpoint = config.backendUrl;


function PDFUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      setUploaded(false);
      setUploadProgress(0);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      toast.error('Please select a file.');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    setIsUploading(true);
    
    try {
      const response = await axios.post(`${apiEndpoint}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: progressEvent => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        }
      });
      toast.success('File uploaded successfully.');
      setUploaded(true);
      navigate(`/chatbot/${response.data.id}`);
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error('Error uploading file.');
      setUploaded(false);
    } finally {
      setIsUploading(false);
      setUploadProgress(null);
    }
  };

  return (
    <Paper className="upload-container" elevation={3}>
      <Typography variant="h4" gutterBottom>
        Upload a Contract PDF
      </Typography>
      <Typography variant="body1" sx={{ mb: 2 }}>
        Click the big circle below to select a PDF file. The title will be automatically derived from the file name.
      </Typography>
      <Box
        component="form"
        onSubmit={handleSubmit}
        className="upload-form"
        sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
      >
        {!selectedFile && (
          <Button
            variant="contained"
            component="label"
            sx={{
              width: 150,
              height: 150,
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 2,
            }}
          >
            <AddCircleOutlineIcon style={{ fontSize: 80, color: 'white' }} />
            <input type="file" hidden accept="application/pdf" onChange={handleFileChange} />
          </Button>
        )}
        {selectedFile && !uploaded && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ mb: 1 }}>{selectedFile.name}</Typography>
            <Button variant="contained" onClick={handleSubmit} disabled={isUploading}>
              {isUploading ? 'Uploading...' : 'Upload Contract'}
            </Button>
          </Box>
        )}
        {isUploading && (
          <Box sx={{ display: 'flex', alignItems: 'center', flexDirection: 'column', mb: 2 }}>
            <CircularProgress variant="determinate" value={uploadProgress} />
            <Typography variant="body1" sx={{ mt: 1 }}>
              {uploadProgress}%
            </Typography>
          </Box>
        )}
        {uploaded && selectedFile && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 2 }}>
            <CheckCircleOutlineIcon style={{ fontSize: 80, color: '#4caf50' }} />
            <Typography variant="h6" sx={{ mt: 1 }}>{selectedFile.name}</Typography>
            <Typography variant="body1" sx={{ mt: 1 }}>File uploaded successfully!</Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
}

export default PDFUpload;
