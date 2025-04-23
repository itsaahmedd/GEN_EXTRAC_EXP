import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  TextField,
  Dialog,
  DialogContent,
  DialogTitle,
  Pagination,
  Alert,
  Container
} from '@mui/material';
import './ListContracts.css';


import config from './config';
const apiEndpoint = config.backendUrl;



function ListContracts() {
  const [contracts, setContracts] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [contractsPerPage] = useState(6);
  const [selectedPdf, setSelectedPdf] = useState(null);

  // Fetch all contracts
  const fetchContracts = () => {
    axios.get(`${apiEndpoint}/contracts`)
    .then(response => {
        setContracts(response.data);
      })
      .catch(error => {
        console.error("Error fetching contracts:", error);
      });
  };

  useEffect(() => {
    fetchContracts();
  }, []);

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1);
  };

  const filteredContracts = contracts.filter(c =>
    c.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const indexOfLastContract = currentPage * contractsPerPage;
  const indexOfFirstContract = indexOfLastContract - contractsPerPage;
  const currentContracts = filteredContracts.slice(indexOfFirstContract, indexOfLastContract);
  const totalPages = Math.ceil(filteredContracts.length / contractsPerPage);

  const handlePageChange = (event, value) => {
    setCurrentPage(value);
  };

  const openPreview = (file_path) => {
    setSelectedPdf(`${file_path}?t=${new Date().getTime()}`);
  };

  const closePreview = () => {
    setSelectedPdf(null);
  };

  // Delete the contract and its chat history.
  const handleDelete = async (contractId) => {
    if (!window.confirm("Are you sure you want to delete this contract and its chat history?")) {
      return;
    }
    try {
      // Delete the contract record
      await axios.delete(`${apiEndpoint}/contracts/${contractId}`);
      // Delete the corresponding chat history (assuming your backend supports this)
      await axios.delete(`${apiEndpoint}/conversations/${contractId}`);
      // Refresh the contracts list
      fetchContracts();
    } catch (error) {
      console.error("Error deleting contract and chat history:", error);
    }
  };

  // Reset all contracts (existing functionality)
  const handleReset = async () => {
    try {
      await axios.delete(`${apiEndpoint}/contracts`);
      fetchContracts(); // Refresh the list
    } catch (error) {
      console.error("Error deleting contracts:", error);
    }
  };

  return (
    <Container className="contracts-container">
      {/* Header Section */}
      <Box className="header-area">
        <Typography variant="h3" className="page-title">
          Your Agreements
        </Typography>
        <Typography variant="subtitle1" className="page-subtitle">
          Manage and review your contracts effortlessly.
        </Typography>
      </Box>

      {/* Search / Reset Section */}
      <Box className="search-area">
        <TextField
          fullWidth
          label="Search Contracts"
          variant="outlined"
          value={searchTerm}
          onChange={handleSearchChange}
        />
        <Button variant="contained" color="error" onClick={handleReset} fullWidth>
          Reset All
        </Button>
      </Box>

      {contracts.length === 0 ? (
        <Box className="no-contracts">
          <Alert severity="info">
            You haven't uploaded any contracts yet.
          </Alert>
          <Typography variant="h6">
            Please upload a contract to get started.
          </Typography>
          <Button variant="contained" color="primary" component={Link} to="/upload">
            Upload Contract
          </Button>
        </Box>
      ) : (
        <>
          {filteredContracts.length === 0 && (
            <Typography variant="body1" align="center" sx={{ mt: 3 }}>
              No matching contracts found for <strong>"{searchTerm}"</strong>.
            </Typography>
          )}

          {/* Contracts List */}
          <Box className="contracts-grid">
            {currentContracts.map(contract => {
              const pdfUrl = `${apiEndpoint}/${contract.file_path}`;
              return (
                <Card key={contract.id} className="contract-card">
                  <CardContent className="card-content">
                    <Typography variant="h6" className="contract-title">
                      {contract.title}
                    </Typography>
                    <Box className="pdf-preview">
                      <embed
                        src={`${pdfUrl}?t=${new Date().getTime()}`}
                        type="application/pdf"
                        width="100%"
                        height="100%"
                      />
                    </Box>
                  </CardContent>
                  <CardActions className="card-actions">
                    <Button variant="contained" onClick={() => openPreview(pdfUrl)} fullWidth>
                      Preview
                    </Button>
                    <Button variant="contained" color="secondary" component={Link} to={`/chatbot/${contract.id}`} fullWidth>
                      Chat
                    </Button>
                    <Button variant="contained" color="error" onClick={() => handleDelete(contract.id)} fullWidth>
                      Delete Contract
                    </Button>
                  </CardActions>
                </Card>
              );
            })}
          </Box>
          {filteredContracts.length > contractsPerPage && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Pagination
                count={totalPages}
                page={currentPage}
                onChange={handlePageChange}
                color="primary"
              />
            </Box>
          )}
        </>
      )}

      {/* PDF Preview Dialog */}
      <Dialog
        open={Boolean(selectedPdf)}
        onClose={closePreview}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>PDF Preview</DialogTitle>
        <DialogContent>
          {selectedPdf && (
            <embed
              key={selectedPdf}
              src={selectedPdf}
              type="application/pdf"
              width="100%"
              height="600px"
            />
          )}
        </DialogContent>
      </Dialog>
    </Container>
  );
}

export default ListContracts;
