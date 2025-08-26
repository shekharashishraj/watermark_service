import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { motion, AnimatePresence } from 'framer-motion';
import { toast, Toaster } from 'react-hot-toast';
import styled from 'styled-components';
import { 
  FileText, 
  Wifi,
  WifiOff
} from 'lucide-react';
import FileUpload from './components/FileUpload';
import WatermarkConfig from './components/WatermarkConfig';
import VisualPositioning from './components/VisualPositioning';
import DownloadSection from './components/DownloadSection';

const API_BASE_URL = 'http://localhost:5001/api';
const WS_URL = 'http://localhost:5001';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
`;

const Header = styled.header`
  text-align: center;
  color: white;
  margin-bottom: 40px;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 10px;
  text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
`;

const Subtitle = styled.p`
  font-size: 1.1rem;
  opacity: 0.9;
  font-weight: 300;
  margin-bottom: 15px;
`;

const ConnectionStatus = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 0.9rem;
  font-weight: 500;
  padding: 8px 16px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  width: fit-content;
  margin: 0 auto;
`;

const MainContent = styled.main`
  max-width: 1200px;
  margin: 0 auto;
`;

const Card = styled(motion.div)`
  background: white;
  border-radius: 16px;
  padding: 30px;
  margin-bottom: 30px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
  }
`;

const LoadingOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  backdrop-filter: blur(5px);
`;

const LoadingContent = styled.div`
  text-align: center;
  color: white;
`;

const Spinner = styled.div`
  width: 50px;
  height: 50px;
  border: 4px solid rgba(255,255,255,0.3);
  border-top: 4px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

function App() {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [currentFileId, setCurrentFileId] = useState(null);
  const [watermarks, setWatermarks] = useState([]);
  const [watermarkCounter, setWatermarkCounter] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('Processing...');
  const [showVisualPositioning, setShowVisualPositioning] = useState(false);
  const [outputFile, setOutputFile] = useState(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io(WS_URL);
    
    newSocket.on('connect', () => {
      console.log('Connected to WebSocket server');
      setIsConnected(true);
      toast.success('Connected to server');
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
      setIsConnected(false);
      toast.error('Disconnected from server');
    });

    newSocket.on('session_joined', (data) => {
      console.log('Joined session:', data);
      setWatermarks(data.watermarks || []);
    });

    newSocket.on('watermark_position_updated', (data) => {
      console.log('Watermark position updated:', data);
      setWatermarks(prev => prev.map(w => 
        w.id === data.watermark_id 
          ? { ...w, position: data.position }
          : w
      ));
    });

    newSocket.on('watermark_added', (data) => {
      console.log('Watermark added:', data);
      // Check if watermark already exists
      setWatermarks(prev => {
        const exists = prev.find(w => w.id === data.watermark.id);
        if (!exists) {
          return [...prev, data.watermark];
        } else {
          console.log('Watermark already exists, skipping duplicate');
          return prev;
        }
      });
    });

    newSocket.on('watermark_removed', (data) => {
      console.log('Watermark removed:', data);
      setWatermarks(prev => prev.filter(w => w.id !== data.watermark_id));
    });

    newSocket.on('watermark_properties_updated', (data) => {
      console.log('Watermark properties updated:', data);
      setWatermarks(prev => prev.map(w => 
        w.id === data.watermark_id 
          ? { ...w, ...data.properties }
          : w
      ));
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const showLoading = (message = 'Processing...') => {
    setLoadingMessage(message);
    setIsLoading(true);
  };

  const hideLoading = () => {
    setIsLoading(false);
  };

  const addWatermark = () => {
    const newCounter = watermarkCounter + 1;
    setWatermarkCounter(newCounter);
    
    const watermark = {
      id: `watermark-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`, // Ensure unique ID
      text: 'CONFIDENTIAL',
      position: 'center',
      font_size: 24,
      color: '#000000',
      opacity: 0.5,
      rotation: 0,
      custom_x: '',
      custom_y: ''
    };

    console.log(`Adding watermark with ID: ${watermark.id}`);

    if (socket && isConnected && currentFileId) {
      socket.emit('add_watermark', {
        file_id: currentFileId,
        watermark: watermark
      });
    }
  };

  const removeWatermark = (watermarkId) => {
    if (socket && isConnected && currentFileId) {
      socket.emit('remove_watermark', {
        file_id: currentFileId,
        watermark_id: watermarkId
      });
    }
  };

  const updateWatermarkProperty = (watermarkId, property, value) => {
    if (socket && isConnected && currentFileId) {
      socket.emit('update_watermark_properties', {
        file_id: currentFileId,
        watermark_id: watermarkId,
        properties: { [property]: value }
      });
    }
  };

  const applyWatermarks = async () => {
    if (!currentFileId || watermarks.length === 0) {
      toast.error('Please upload a file and add watermarks first.');
      return;
    }

    showLoading('Applying watermarks...');

    try {
      const response = await fetch(`${API_BASE_URL}/watermark`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file_id: currentFileId,
          watermarks: watermarks
        })
      });

      const data = await response.json();

      if (data.success) {
        setOutputFile(data.output_file);
        toast.success(data.message || 'Watermarks applied successfully!');
      } else {
        toast.error(data.error || 'Failed to apply watermarks.');
      }
    } catch (error) {
      toast.error('Failed to apply watermarks. Please try again.');
      console.error('Watermark error:', error);
    } finally {
      hideLoading();
    }
  };

  const downloadFile = () => {
    if (outputFile) {
      window.open(`${API_BASE_URL}/download/${outputFile}`, '_blank');
    }
  };

  const cleanupFiles = async () => {
    if (!currentFileId) return;

    try {
      await fetch(`${API_BASE_URL}/cleanup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file_id: currentFileId })
      });
    } catch (error) {
      console.error('Cleanup error:', error);
    }
  };

  return (
    <AppContainer>
      <Toaster position="top-right" />
      
      <Header>
        <Title>
          <FileText size={40} />
          PDF Watermark Service
        </Title>
        <Subtitle>
          Professional PDF watermarking with real-time drag-and-drop positioning
        </Subtitle>
        <ConnectionStatus>
          {isConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </ConnectionStatus>
      </Header>

      <MainContent>
        <AnimatePresence>
          {!currentFileId && (
            <Card
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <FileUpload
                onFileUploaded={(fileId, fileName) => {
                  setCurrentFileId(fileId);
                  if (socket && isConnected) {
                    socket.emit('join_session', { file_id: fileId });
                  }
                }}
                apiBaseUrl={API_BASE_URL}
                showLoading={showLoading}
                hideLoading={hideLoading}
              />
            </Card>
          )}

          {currentFileId && (
            <>
              <Card
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <WatermarkConfig
                  watermarks={watermarks}
                  onAddWatermark={addWatermark}
                  onRemoveWatermark={removeWatermark}
                  onUpdateProperty={updateWatermarkProperty}
                  onApplyWatermarks={applyWatermarks}
                  onShowVisualPositioning={() => setShowVisualPositioning(true)}
                />
              </Card>

              {showVisualPositioning && (
                <Card
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <VisualPositioning
                    watermarks={watermarks}
                    onUpdateProperty={updateWatermarkProperty}
                    onRemoveWatermark={removeWatermark}
                    onClose={() => setShowVisualPositioning(false)}
                    currentFileId={currentFileId}
                    apiBaseUrl={API_BASE_URL}
                  />
                </Card>
              )}

              {outputFile && (
                <Card
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <DownloadSection
                    outputFile={outputFile}
                    onDownload={downloadFile}
                    onReset={() => {
                      setCurrentFileId(null);
                      setWatermarks([]);
                      setWatermarkCounter(0);
                      setOutputFile(null);
                      setShowVisualPositioning(false);
                      cleanupFiles();
                    }}
                  />
                </Card>
              )}
            </>
          )}
        </AnimatePresence>
      </MainContent>

      <AnimatePresence>
        {isLoading && (
          <LoadingOverlay
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <LoadingContent>
              <Spinner />
              <p>{loadingMessage}</p>
            </LoadingContent>
          </LoadingOverlay>
        )}
      </AnimatePresence>
    </AppContainer>
  );
}

export default App;
