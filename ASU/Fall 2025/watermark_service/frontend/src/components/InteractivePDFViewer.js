import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import { 
  ZoomIn, ZoomOut, FileText, Maximize2, Minimize2, Grid,
  MousePointer2, Hand, Trash2, RotateCw, RefreshCw
} from 'lucide-react';

// Set up PDF.js worker - using a CDN that works with Create React App
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

const ViewerContainer = styled.div`
  position: relative;
  background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 20px;
  padding: 25px;
  box-shadow: 
    0 20px 40px rgba(0,0,0,0.1),
    inset 0 1px 0 rgba(255,255,255,0.8);
  min-height: 700px;
`;

const Toolbar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  margin-bottom: 20px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.1);
  border: 1px solid rgba(255,255,255,0.2);
`;

const ToolbarGroup = styled.div`
  display: flex;
  gap: 10px;
  align-items: center;
`;

const ToolbarButton = styled(motion.button)`
  padding: 10px;
  border: none;
  border-radius: 10px;
  background: ${props => props.active ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#f7fafc'};
  color: ${props => props.active ? 'white' : '#4a5568'};
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;

  &:hover {
    background: ${props => props.active ? 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)' : '#edf2f7'};
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
  }

  &:active {
    transform: translateY(0px);
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.6s;
  }

  &:hover::before {
    left: 100%;
  }
`;

const ZoomControls = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  background: rgba(255,255,255,0.9);
  padding: 8px;
  border-radius: 12px;
  backdrop-filter: blur(10px);
`;

const ZoomSlider = styled.input`
  width: 100px;
  height: 4px;
  border-radius: 5px;
  background: #e2e8f0;
  outline: none;
  
  &::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  }
`;

const PDFContainer = styled.div`
  position: relative;
  background: #ffffff;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 
    0 15px 35px rgba(0,0,0,0.1),
    0 5px 15px rgba(0,0,0,0.05);
  border: 1px solid rgba(0,0,0,0.05);
  min-height: 600px;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const PDFWrapper = styled.div`
  position: relative;
  transform: scale(${props => props.scale});
  transform-origin: center center;
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: ${props => props.dragMode ? 'grab' : 'crosshair'};
  
  &:active {
    cursor: ${props => props.dragMode ? 'grabbing' : 'crosshair'};
  }
`;

const WatermarkOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
`;

const InteractiveWatermark = styled(motion.div)`
  position: absolute;
  cursor: grab;
  user-select: none;
  pointer-events: all;
  z-index: 20;
  
  &:active {
    cursor: grabbing;
  }
`;

const WatermarkContent = styled(motion.div)`
  padding: 8px 16px;
  border-radius: 8px;
  font-weight: 600;
  font-size: ${props => props.fontSize}px;
  color: ${props => props.color};
  opacity: ${props => props.opacity};
  transform: rotate(${props => props.rotation}deg);
  background: ${props => props.showBackground ? 'rgba(255,255,255,0.9)' : 'transparent'};
  border: 2px solid ${props => props.isSelected ? '#667eea' : 'transparent'};
  box-shadow: ${props => props.isSelected ? '0 0 20px rgba(102, 126, 234, 0.5)' : '0 4px 12px rgba(0,0,0,0.15)'};
  backdrop-filter: blur(5px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    border: 2px dashed rgba(102, 126, 234, 0.5);
    border-radius: 12px;
    opacity: ${props => props.isSelected ? 1 : 0};
    transition: opacity 0.3s ease;
  }
`;

const WatermarkControls = styled(motion.div)`
  position: absolute;
  top: -40px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 5px;
  background: rgba(0,0,0,0.8);
  padding: 5px 8px;
  border-radius: 8px;
  backdrop-filter: blur(10px);
`;

const WatermarkControlButton = styled.button`
  padding: 4px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255,255,255,0.2);
  }
`;

const GridOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: ${props => props.show ? 0.3 : 0};
  transition: opacity 0.3s ease;
  background-image: 
    linear-gradient(rgba(102, 126, 234, 0.5) 1px, transparent 1px),
    linear-gradient(90deg, rgba(102, 126, 234, 0.5) 1px, transparent 1px);
  background-size: 20px 20px;
  pointer-events: none;
  z-index: 5;
`;

const StatusBar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  margin-top: 15px;
  font-size: 0.9rem;
  color: #4a5568;
  border: 1px solid rgba(0,0,0,0.05);
`;

const CoordinateDisplay = styled.div`
  font-family: 'Monaco', 'Menlo', monospace;
  background: #f7fafc;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 0.8rem;
`;

const LoadingSpinner = styled(motion.div)`
  width: 60px;
  height: 60px;
  border: 4px solid #e2e8f0;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  margin: 0 auto 20px;
  
  animation: spin 1s linear infinite;
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

function InteractivePDFViewer({ 
  currentFileId, 
  apiBaseUrl, 
  watermarks, 
  onUpdateWatermark,
  onRemoveWatermark,
  selectedWatermark,
  onSelectWatermark
}) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [isDragMode, setIsDragMode] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [draggedWatermark, setDraggedWatermark] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [pdfLoading, setPdfLoading] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const pdfRef = useRef(null);
  const containerRef = useRef(null);

  const pdfUrl = currentFileId ? `${apiBaseUrl}/original/${currentFileId}` : null;

  const onDocumentLoadSuccess = useCallback(({ numPages }) => {
    setNumPages(numPages);
    setPdfLoading(false);
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePosition({ x: Math.round(x), y: Math.round(y) });

    if (draggedWatermark && pdfRef.current) {
      const pdfRect = pdfRef.current.getBoundingClientRect();
      
      const relativeX = (e.clientX - pdfRect.left) / scale;
      const relativeY = (e.clientY - pdfRect.top) / scale;
      
      // Convert to PDF coordinates (assuming standard PDF size)
      const pdfX = (relativeX / pdfRect.width) * 595;
      const pdfY = (relativeY / pdfRect.height) * 842;
      
      onUpdateWatermark(draggedWatermark, {
        position: 'custom',
        custom_x: Math.round(pdfX),
        custom_y: Math.round(pdfY)
      });
    }
  }, [draggedWatermark, scale, onUpdateWatermark]);

  const handleWatermarkMouseDown = useCallback((e, watermarkId) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggedWatermark(watermarkId);
    onSelectWatermark(watermarkId);
  }, [onSelectWatermark]);

  const handleMouseUp = useCallback(() => {
    setDraggedWatermark(null);
  }, []);

  useEffect(() => {
    if (draggedWatermark) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [draggedWatermark, handleMouseMove, handleMouseUp]);

  const getWatermarkPosition = (watermark) => {
    if (!pdfRef.current) return { x: 100, y: 100 };
    
    const pdfRect = pdfRef.current.getBoundingClientRect();
    
    if (watermark.position === 'custom' && watermark.custom_x && watermark.custom_y) {
      const x = (watermark.custom_x / 595) * pdfRect.width * scale;
      const y = (watermark.custom_y / 842) * pdfRect.height * scale;
      return { x, y };
    }
    
    // Default positions
    const centerX = pdfRect.width * scale / 2;
    const centerY = pdfRect.height * scale / 2;
    
    const positions = {
      'center': { x: centerX, y: centerY },
      'top-left': { x: 50, y: 50 },
      'top-center': { x: centerX, y: 50 },
      'top-right': { x: pdfRect.width * scale - 50, y: 50 },
      'bottom-center': { x: centerX, y: pdfRect.height * scale - 50 }
    };
    
    return positions[watermark.position] || positions.center;
  };

  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));
  const resetZoom = () => setScale(1.0);

  return (
    <ViewerContainer ref={containerRef}>
      <Toolbar>
        <ToolbarGroup>
          <ToolbarButton 
            active={isDragMode}
            onClick={() => setIsDragMode(!isDragMode)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isDragMode ? <Hand size={18} /> : <MousePointer2 size={18} />}
          </ToolbarButton>
          
          <ToolbarButton 
            active={showGrid}
            onClick={() => setShowGrid(!showGrid)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Grid size={18} />
          </ToolbarButton>
          
          <ToolbarButton 
            onClick={() => setIsFullscreen(!isFullscreen)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isFullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </ToolbarButton>
        </ToolbarGroup>

        <ZoomControls>
          <ToolbarButton onClick={zoomOut} whileHover={{ scale: 1.1 }}>
            <ZoomOut size={16} />
          </ToolbarButton>
          
          <ZoomSlider
            type="range"
            min="0.5"
            max="3.0"
            step="0.1"
            value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
          />
          
          <ToolbarButton onClick={zoomIn} whileHover={{ scale: 1.1 }}>
            <ZoomIn size={16} />
          </ToolbarButton>
          
          <ToolbarButton onClick={resetZoom} whileHover={{ scale: 1.1 }}>
            <RefreshCw size={16} />
          </ToolbarButton>
        </ZoomControls>

        <ToolbarGroup>
          <CoordinateDisplay>
            X: {mousePosition.x} Y: {mousePosition.y} | Zoom: {Math.round(scale * 100)}%
          </CoordinateDisplay>
        </ToolbarGroup>
      </Toolbar>

      <PDFContainer>
        <GridOverlay show={showGrid} />
        
        {pdfLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ textAlign: 'center', padding: '60px' }}
          >
            <LoadingSpinner />
            <p>Loading interactive PDF viewer...</p>
          </motion.div>
        )}
        
        {pdfUrl && (
          <PDFWrapper 
            ref={pdfRef}
            scale={scale}
            dragMode={isDragMode}
          >
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={
                <motion.div
                  initial={{ scale: 0.9 }}
                  animate={{ scale: 1 }}
                  style={{ textAlign: 'center', padding: '40px' }}
                >
                  <LoadingSpinner />
                  <p>Rendering PDF...</p>
                </motion.div>
              }
              error={
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <FileText size={48} color="#cbd5e0" />
                  <p>Failed to load PDF. Using drag interface.</p>
                </div>
              }
            >
              <Page
                pageNumber={pageNumber}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            </Document>
            
            <WatermarkOverlay>
              <AnimatePresence>
                {watermarks.map((watermark) => {
                  const position = getWatermarkPosition(watermark);
                  const isSelected = selectedWatermark === watermark.id;
                  
                  return (
                    <InteractiveWatermark
                      key={watermark.id}
                      style={{
                        left: position.x,
                        top: position.y,
                        transform: 'translate(-50%, -50%)'
                      }}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0, opacity: 0 }}
                      onMouseDown={(e) => handleWatermarkMouseDown(e, watermark.id)}
                      onClick={() => onSelectWatermark(watermark.id)}
                      whileHover={{ scale: 1.05 }}
                      whileDrag={{ scale: 1.1, zIndex: 100 }}
                    >
                      <WatermarkContent
                        fontSize={watermark.font_size}
                        color={watermark.color}
                        opacity={watermark.opacity}
                        rotation={watermark.rotation}
                        isSelected={isSelected}
                        showBackground={isSelected}
                        whileHover={{ 
                          boxShadow: '0 8px 25px rgba(102, 126, 234, 0.3)' 
                        }}
                      >
                        {watermark.text}
                      </WatermarkContent>
                      
                      <AnimatePresence>
                        {isSelected && (
                          <WatermarkControls
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 10 }}
                          >
                            <WatermarkControlButton
                              onClick={(e) => {
                                e.stopPropagation();
                                onUpdateWatermark(watermark.id, { 
                                  rotation: (watermark.rotation + 15) % 360 
                                });
                              }}
                            >
                              <RotateCw size={12} />
                            </WatermarkControlButton>
                            
                            <WatermarkControlButton
                              onClick={(e) => {
                                e.stopPropagation();
                                onRemoveWatermark(watermark.id);
                              }}
                            >
                              <Trash2 size={12} />
                            </WatermarkControlButton>
                          </WatermarkControls>
                        )}
                      </AnimatePresence>
                    </InteractiveWatermark>
                  );
                })}
              </AnimatePresence>
            </WatermarkOverlay>
          </PDFWrapper>
        )}
      </PDFContainer>

      <StatusBar>
        <div>
          {numPages && `Page ${pageNumber} of ${numPages}`} | {watermarks.length} watermarks
        </div>
        <div>
          {draggedWatermark ? 'Dragging watermark...' : 'Click and drag watermarks to position them'}
        </div>
      </StatusBar>
    </ViewerContainer>
  );
}

export default InteractivePDFViewer;