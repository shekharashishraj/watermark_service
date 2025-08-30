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
  height: 100vh;
  background: #f8fafc;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  display: flex;
  overflow: hidden;
`;

const Sidebar = styled(motion.aside)`
  width: 420px;
  background: white;
  border-right: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 2px 0 10px rgba(0,0,0,0.05);
`;

const SidebarHeader = styled.div`
  padding: 24px;
  border-bottom: 1px solid #e2e8f0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
  gap: 12px;
`;

const Subtitle = styled.p`
  font-size: 0.9rem;
  opacity: 0.9;
  margin: 0 0 12px 0;
  font-weight: 300;
`;

const ConnectionStatus = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  font-weight: 500;
  padding: 6px 12px;
  border-radius: 15px;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  width: fit-content;
`;

const SidebarContent = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 0;
`;

const MainPanel = styled.main`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
  overflow: hidden;
`;

const PanelHeader = styled.div`
  padding: 16px 24px;
  background: white;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: between;
  align-items: center;
  min-height: 60px;
`;

const PanelTitle = styled.h2`
  font-size: 1.1rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0;
`;

const PreviewContainer = styled.div`
  flex: 1;
  overflow: hidden;
  position: relative;
  background: #f1f5f9;
`;

const Section = styled.div`
  border-bottom: 1px solid #e2e8f0;
  
  &:last-child {
    border-bottom: none;
  }
`;

const SectionHeader = styled.div`
  padding: 16px 24px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  font-weight: 600;
  font-size: 0.9rem;
  color: #4a5568;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const SectionContent = styled.div`
  padding: 24px;
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
  const [outputFile, setOutputFile] = useState(null);
  const [pdfInfo, setPdfInfo] = useState(null);

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
      
      <Sidebar
        initial={{ x: -420 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
      >
        <SidebarHeader>
          <Title>
            <FileText size={24} />
            Watermark Studio
          </Title>
          <Subtitle>
            Professional PDF watermarking tool
          </Subtitle>
          <ConnectionStatus>
            {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </ConnectionStatus>
        </SidebarHeader>

        <SidebarContent>
          {!currentFileId && (
            <Section>
              <SectionHeader>
                <FileText size={16} />
                Upload Document
              </SectionHeader>
              <SectionContent>
                <FileUpload
                  onFileUploaded={async (fileId, fileName) => {
                    setCurrentFileId(fileId);
                    
                    // Fetch PDF information
                    try {
                      const response = await fetch(`${API_BASE_URL}/pdf-info/${fileId}`);
                      if (response.ok) {
                        const pdfData = await response.json();
                        setPdfInfo(pdfData);
                        console.log('PDF Info:', pdfData);
                      }
                    } catch (error) {
                      console.error('Failed to fetch PDF info:', error);
                    }
                    
                    if (socket && isConnected) {
                      socket.emit('join_session', { file_id: fileId });
                    }
                  }}
                  apiBaseUrl={API_BASE_URL}
                  showLoading={showLoading}
                  hideLoading={hideLoading}
                />
              </SectionContent>
            </Section>
          )}

          {currentFileId && (
            <>
              <Section>
                <WatermarkConfig
                  watermarks={watermarks}
                  onAddWatermark={addWatermark}
                  onRemoveWatermark={removeWatermark}
                  onUpdateProperty={updateWatermarkProperty}
                  onApplyWatermarks={applyWatermarks}
                  onShowVisualPositioning={() => {}}
                  pdfInfo={pdfInfo}
                  isCompact={true}
                />
              </Section>

              {outputFile && (
                <Section>
                  <SectionHeader>
                    <FileText size={16} />
                    Download Result
                  </SectionHeader>
                  <SectionContent>
                    <DownloadSection
                      outputFile={outputFile}
                      onDownload={downloadFile}
                      onReset={() => {
                        setCurrentFileId(null);
                        setWatermarks([]);
                        setWatermarkCounter(0);
                        setOutputFile(null);
                        cleanupFiles();
                      }}
                      isCompact={true}
                    />
                  </SectionContent>
                </Section>
              )}
            </>
          )}
        </SidebarContent>
      </Sidebar>

      <MainPanel>
        {currentFileId ? (
          <>
            <PanelHeader>
              <PanelTitle>Document Preview & Watermark Positioning</PanelTitle>
            </PanelHeader>
            <PreviewContainer>
              <VisualPositioning
                watermarks={watermarks}
                onUpdateProperty={updateWatermarkProperty}
                onRemoveWatermark={removeWatermark}
                onClose={() => {}}
                currentFileId={currentFileId}
                apiBaseUrl={API_BASE_URL}
                pdfInfo={pdfInfo}
                isIntegrated={true}
              />
            </PreviewContainer>
          </>
        ) : (
          <div style={{ 
            flex: 1, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            color: '#718096',
            fontSize: '1.1rem',
            textAlign: 'center'
          }}>
            <div>
              <FileText size={64} style={{ marginBottom: '16px', opacity: 0.5 }} />
              <div>Upload a PDF document to begin watermarking</div>
            </div>
          </div>
        )}
      </MainPanel>

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
