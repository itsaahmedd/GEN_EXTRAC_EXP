import React, { useState, useEffect, useCallback} from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import config from './config';

import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemText,
  Tabs,
  Tab,
  Typography,
  Button,
  TextField,
  Paper,
  Link
} from '@mui/material';
import './ChatbotLayout.css';

function TabPanel({ children, value, index }) {
  return (
    <div
      role="tabpanel"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        overflow: 'auto',
        visibility: value === index ? 'visible' : 'hidden',
        pointerEvents: value === index ? 'auto' : 'none',
        transition: 'opacity 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
    >
      {children}
    </div>
  );
}

const apiEndpoint = config.backendUrl;
const chatbotEndpoint = config.ngrokUrl; // Use this instead of hardcoding



const ChatbotLayout = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [contracts, setContracts] = useState([]);
  // Store conversations by contract ID
  const [conversations, setConversations] = useState({});
  const [selectedContract, setSelectedContract] = useState(null);
  const [pdfUrl, setPdfUrl] = useState('');
  const [tabIndex, setTabIndex] = useState(0); // 0: Chat, 1: PDF Preview
  const [question, setQuestion] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  // Fetch conversation for a given contract ID from the backend.
  const fetchConversation = async (contractId) => {
    try {
      const response = await axios.get(`${apiEndpoint}/conversations/${contractId}`);
      // Save conversation for this contract
      setConversations(prev => ({ ...prev, [contractId]: response.data.messages || [] }));
    } catch (error) {
      console.error("Error fetching conversation:", error);
      setConversations(prev => ({ ...prev, [contractId]: [] }));
    }
  };

  // Save conversation for a given contract ID on the backend.
  const saveConversation = async (contractId, messages) => {
    try {
      await axios.post(`${apiEndpoint}/conversations`, {
        contract_id: contractId,
        messages: messages
      });
    } catch (error) {
      console.error("Error saving conversation:", error);
    }
  };

  
  const handleContractSelect = useCallback((contract) => {
    setSelectedContract(contract);
    setQuestion("");
    setTabIndex(0);
    const url = `${config.backendUrl}/${contract.file_path}?t=${Date.now()}`;
    setPdfUrl(url);
    navigate(`/chatbot/${contract.id}`);
    fetchConversation(contract.id);
  }, [navigate]);


  useEffect(() => {
    axios.get(`${config.backendUrl}/contracts`)
      .then(response => {
        setContracts(response.data);
        const initialContract = response.data.find(c => c.id.toString() === id) || response.data[0];
        if (initialContract) {
          handleContractSelect(initialContract);
        }
      })
      .catch(error => console.error("Error fetching contracts:", error));
  }, [id, handleContractSelect]);
  
  

  const handleTabChange = (event, newValue) => {
    setTabIndex(newValue);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    // Get the conversation for the current contract (or initialize it if empty)
    const currentConversation = conversations[selectedContract.id] || [];
    const updatedConversation = [...currentConversation, { sender: 'user', text: question }];
    
    // Update conversation state for this contract
    setConversations(prev => ({ ...prev, [selectedContract.id]: updatedConversation }));
    await saveConversation(selectedContract.id, updatedConversation);
    setQuestion("");
    setIsGenerating(true);

    try {
      // 1. Retrieve context from local backend.
      const retrievalResponse = await axios.post(`${apiEndpoint}/retrieve-context`, {
        contract_id: selectedContract.id,
        question: question
      });
      const retrievedContext = retrievalResponse.data.context || "";
      
      // 2. Use only the last 3 messages from the conversation for this contract.
      const conversationForPrompt = updatedConversation.slice(-3);
      const formattedChatHistory = conversationForPrompt
        .map(msg => msg.sender === 'user' ? `User: ${msg.text}` : `Bot: ${msg.text}`)
        .join("\n");
      
      // 3. Call Colab inference endpoint with question, context, and contract-specific conversation.
      const genResponse = await axios.post(`${chatbotEndpoint}/generate-answer`, {
        question: question,
        context: retrievedContext,
        chatHistory: formattedChatHistory
      });
      const botAnswer = genResponse.data.answer || "This is a placeholder answer.";
      
      const finalConversation = [...updatedConversation, { sender: 'bot', text: botAnswer }];
      setConversations(prev => ({ ...prev, [selectedContract.id]: finalConversation }));
      await saveConversation(selectedContract.id, finalConversation);
    } catch (error) {
      console.error("Error fetching bot response", error);
      const errorConversation = [...updatedConversation, { sender: 'bot', text: "Error retrieving answer." }];
      setConversations(prev => ({ ...prev, [selectedContract.id]: errorConversation }));
      await saveConversation(selectedContract.id, errorConversation);
    } finally {
      setIsGenerating(false);
    }
  };

  // Get the conversation for the currently selected contract.
  const currentChatHistory = selectedContract ? (conversations[selectedContract.id] || []) : [];

  return (
    <Box className="chatbot-layout">
      {/* Left Sidebar */}
      <Drawer
        variant="permanent"
        anchor="left"
        sx={{
          width: 250,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 250,
            backgroundColor: '#6A1B9A',
            color: '#FFFFFF',
          },
        }}
      >
        <Box sx={{ overflow: 'auto' }}>
          <Typography variant="h6" sx={{ p: 2, borderBottom: '1px solid rgba(255,255,255,0.3)' }}>
            Contracts
          </Typography>
          <List>
            {contracts.map(contract => (
              <ListItem
                button
                key={contract.id}
                selected={selectedContract && contract.id === selectedContract.id}
                onClick={() => handleContractSelect(contract)}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: '#FFD54F !important',
                    color: '#6A1B9A !important',
                  },
                  '&.MuiListItem-button:hover': {
                    backgroundColor: 'rgba(255,255,255,0.2)',
                  },
                }}
              >
                <ListItemText primary={contract.title} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      {/* Right Panel */}
      <Box className="chatbot-right">
        {selectedContract ? (
          <>
            <Typography variant="h4" align="center" gutterBottom>
              {selectedContract.title}
            </Typography>
            <Tabs value={tabIndex} onChange={handleTabChange} centered>
              <Tab label="Chat" />
              <Tab label="PDF Preview" />
            </Tabs>

            <Box sx={{ flex: 1, position: 'relative', minHeight: 0 }}>
              {/* Chat Panel */}
              <TabPanel value={tabIndex} index={0}>
                <Paper sx={{ p: 2, flex: 1, overflow: 'auto', mb: 2 }}>
                  {currentChatHistory.map((msg, index) => (
                    <Box
                      key={index}
                      sx={{
                        mb: 1,
                        p: 1,
                        borderRadius: 1,
                        maxWidth: '80%',
                        alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                        bgcolor: msg.sender === 'user' ? '#D1C4E9' : '#FFE0B2',
                      }}
                    >
                      <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                        {msg.text}
                      </Typography>
                    </Box>
                  ))}
                  {isGenerating && (
                    <Box sx={{ mt: 1, fontStyle: 'italic', color: 'gray' }}>
                      Bot is typing<span className="dot-animation">...</span>
                    </Box>
                  )}
                </Paper>
                <Box component="form" onSubmit={handleSend} sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    label="Your Question"
                    variant="outlined"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                  />
                  <Button type="submit" variant="contained">
                    Send
                  </Button>
                </Box>
              </TabPanel>

              {/* PDF Preview Panel */}
              <TabPanel value={tabIndex} index={1}>
                <Paper sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <Box sx={{ flex: 1, overflow: 'auto' }}>
                    <embed
                      key={pdfUrl}
                      src={pdfUrl}
                      type="application/pdf"
                      width="100%"
                      height="100%"
                    />
                  </Box>
                  <Box sx={{ textAlign: 'center', mt: 1 }}>
                    <Link href={pdfUrl} target="_blank" rel="noopener" underline="hover">
                      View Full PDF
                    </Link>
                  </Box>
                </Paper>
              </TabPanel>
            </Box>
          </>
        ) : (
          <Typography variant="h6" align="center" sx={{ mt: 4 }}>
            No contract selected.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default ChatbotLayout;
