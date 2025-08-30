import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import { 
  ZoomIn, ZoomOut, FileText, Maximize2, Minimize2, Grid,
  MousePointer2, Hand, Trash2, RotateCw, RefreshCw, Check, X
} from 'lucide-react';

const ViewerContainer = styled.div`
  position: relative;
  background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: ${props => props.isFullscreen ? '15px' : '20px'};
  padding: ${props => props.isFullscreen ? '15px' : '25px'};
  box-shadow: 
    0 20px 40px rgba(0,0,0,0.1),
    inset 0 1px 0 rgba(255,255,255,0.8);
  min-height: ${props => props.isFullscreen ? '80vh' : '700px'};
  max-height: ${props => props.isFullscreen ? '80vh' : 'none'};
  overflow: ${props => props.isFullscreen ? 'hidden' : 'visible'};
  display: flex;
  flex-direction: column;
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

  &:hover {
    background: ${props => props.active ? 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)' : '#edf2f7'};
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
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
  width: 100%;
  height: 600px;
  
  &:active {
    cursor: ${props => props.dragMode ? 'grabbing' : 'crosshair'};
  }
`;

const PDFPlaceholder = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #718096;
  font-size: 1.1rem;
  text-align: center;
  padding: 40px;
  background: 
    linear-gradient(45deg, #f8fafc 25%, transparent 25%), 
    linear-gradient(-45deg, #f8fafc 25%, transparent 25%), 
    linear-gradient(45deg, transparent 75%, #f8fafc 75%), 
    linear-gradient(-45deg, transparent 75%, #f8fafc 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
`;

const PDFPreview = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
  background: white;
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

const LockedWatermarkOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 15;
  transform: scale(${props => props.scale});
  transform-origin: top left;
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
  background: ${props => props.isLocked ? 'rgba(34, 197, 94, 0.1)' : props.isSelected ? 'rgba(255,255,255,0.95)' : 'rgba(102, 126, 234, 0.1)'};
  border: 2px solid ${props => props.isLocked ? '#22c55e' : props.isSelected ? '#667eea' : 'rgba(102, 126, 234, 0.3)'};
  box-shadow: ${props => props.isLocked ? '0 0 20px rgba(34, 197, 94, 0.5)' : props.isSelected ? '0 0 20px rgba(102, 126, 234, 0.5)' : '0 4px 12px rgba(0,0,0,0.15)'};
  backdrop-filter: blur(5px);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  min-width: 120px;
  text-align: center;
  
  &::after {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    border: 2px dashed ${props => props.isLocked ? 'rgba(34, 197, 94, 0.5)' : 'rgba(102, 126, 234, 0.5)'};
    border-radius: 12px;
    opacity: ${props => props.isSelected || props.isLocked ? 1 : 0};
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

// Throttle utility function
const throttle = (func, limit) => {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
};

// Debounce utility function  
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

function SmoothPDFViewer({ 
  currentFileId, 
  apiBaseUrl, 
  watermarks, 
  onUpdateWatermark,
  onRemoveWatermark,
  selectedWatermark,
  onSelectWatermark,
  isFullscreenMode = false
}) {
  // Add state for locked watermarks and their absolute positions
  const [lockedWatermarks, setLockedWatermarks] = useState(new Set());
  const [lockedWatermarkPositions, setLockedWatermarkPositions] = useState({});
  const [scale, setScale] = useState(1.0);
  const [isDragMode, setIsDragMode] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [draggedWatermark, setDraggedWatermark] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [internalFullscreen, setInternalFullscreen] = useState(false);
  const isFullscreen = isFullscreenMode || internalFullscreen;
  
  // Local state for smooth dragging (separate from server state)
  const [localWatermarkPositions, setLocalWatermarkPositions] = useState({});
  
  const pdfRef = useRef(null);
  const containerRef = useRef(null);
  const dragStartPos = useRef({ x: 0, y: 0 });
  const lastUpdateTime = useRef(0);

  const pdfUrl = currentFileId ? `${apiBaseUrl}/original/${currentFileId}` : null;

  // Throttled mouse position update for UI only
  const updateMousePosition = throttle((x, y) => {
    setMousePosition({ x: Math.round(x), y: Math.round(y) });
  }, 16); // ~60fps

  // Debounced server update - only send to server after drag stops
  const debouncedServerUpdate = debounce((watermarkId, position) => {
    console.log('Sending final position to server:', watermarkId, position);
    onUpdateWatermark(watermarkId, position);
  }, 300);

  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    updateMousePosition(x, y);

    if (draggedWatermark && pdfRef.current) {
      const pdfRect = pdfRef.current.getBoundingClientRect();
      
      const relativeX = (e.clientX - pdfRect.left) / scale;
      const relativeY = (e.clientY - pdfRect.top) / scale;
      
      // Convert to PDF coordinates
      const pdfX = Math.max(0, Math.min((relativeX / pdfRect.width) * 595, 595));
      const pdfY = Math.max(0, Math.min((relativeY / pdfRect.height) * 842, 842));
      
      // Update LOCAL state immediately for smooth dragging
      setLocalWatermarkPositions(prev => ({
        ...prev,
        [draggedWatermark]: {
          position: 'custom',
          custom_x: Math.round(pdfX),
          custom_y: Math.round(pdfY)
        }
      }));
      
      // Throttled server update (much less frequent)
      const now = Date.now();
      if (now - lastUpdateTime.current > 100) { // Max 10 updates per second during drag
        lastUpdateTime.current = now;
        debouncedServerUpdate(draggedWatermark, {
          position: 'custom',
          custom_x: Math.round(pdfX),
          custom_y: Math.round(pdfY)
        });
      }
    }
  }, [draggedWatermark, scale, debouncedServerUpdate, updateMousePosition]);

  const handleWatermarkMouseDown = useCallback((e, watermarkId) => {
    // Prevent dragging if watermark is locked
    if (lockedWatermarks.has(watermarkId)) {
      return;
    }
    
    e.preventDefault();
    e.stopPropagation();
    
    setDraggedWatermark(watermarkId);
    onSelectWatermark(watermarkId);
    
    // Store initial drag position
    dragStartPos.current = {
      x: e.clientX,
      y: e.clientY
    };
  }, [onSelectWatermark, lockedWatermarks]);

  const handleMouseUp = useCallback(() => {
    if (draggedWatermark) {
      // Send final position to server
      const localPos = localWatermarkPositions[draggedWatermark];
      if (localPos) {
        console.log('Final drag position update:', draggedWatermark, localPos);
        onUpdateWatermark(draggedWatermark, localPos);
      }
    }
    setDraggedWatermark(null);
  }, [draggedWatermark, localWatermarkPositions, onUpdateWatermark]);

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
    
    // Use local position if dragging, otherwise use server state
    const localPos = localWatermarkPositions[watermark.id];
    const effectiveWatermark = localPos ? { ...watermark, ...localPos } : watermark;
    
    if (effectiveWatermark.position === 'custom' && effectiveWatermark.custom_x && effectiveWatermark.custom_y) {
      const x = (effectiveWatermark.custom_x / 595) * (pdfRect.width / scale);
      const y = (effectiveWatermark.custom_y / 842) * (pdfRect.height / scale);
      return { x, y };
    }
    
    // Default positions
    const centerX = (pdfRect.width / scale) / 2;
    const centerY = (pdfRect.height / scale) / 2;
    
    const positions = {
      'center': { x: centerX, y: centerY },
      'top-left': { x: 50, y: 50 },
      'top-center': { x: centerX, y: 50 },
      'top-right': { x: (pdfRect.width / scale) - 50, y: 50 },
      'bottom-center': { x: centerX, y: (pdfRect.height / scale) - 50 }
    };
    
    return positions[effectiveWatermark.position] || positions.center;
  };

  // Function to get absolute PDF coordinates for locked watermarks
  const getAbsolutePDFPosition = (watermark) => {
    if (!pdfRef.current) return { x: 100, y: 100 };
    
    const pdfRect = pdfRef.current.getBoundingClientRect();
    const position = getWatermarkPosition(watermark);
    
    // Convert to absolute PDF coordinates (0-595 x 0-842 for A4)
    const pdfX = (position.x / (pdfRect.width / scale)) * 595;
    const pdfY = (position.y / (pdfRect.height / scale)) * 842;
    
    return { pdfX, pdfY };
  };

  // Function to get position for locked watermarks using stored absolute coordinates
  const getLockedWatermarkPosition = (watermark) => {
    if (!pdfRef.current || !lockedWatermarkPositions[watermark.id]) {
      return getWatermarkPosition(watermark);
    }
    
    const pdfRect = pdfRef.current.getBoundingClientRect();
    const absolutePos = lockedWatermarkPositions[watermark.id];
    
    // For locked watermarks, we want to position them relative to the PDF content
    // without the scale transformation since the LockedWatermarkOverlay handles scaling
    const x = (absolutePos.pdfX / 595) * pdfRect.width;
    const y = (absolutePos.pdfY / 842) * pdfRect.height;
    
    return { x, y };
  };

  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));
  const resetZoom = () => setScale(1.0);

  // Function to lock watermark position
  const lockWatermark = (watermarkId) => {
    const watermark = watermarks.find(w => w.id === watermarkId);
    if (!watermark) return;
    
    // Get absolute PDF coordinates
    const absolutePos = getAbsolutePDFPosition(watermark);
    
    // Store the locked position
    setLockedWatermarkPositions(prev => ({
      ...prev,
      [watermarkId]: absolutePos
    }));
    
    setLockedWatermarks(prev => new Set([...prev, watermarkId]));
    
    // Clear any local position updates for this watermark
    setLocalWatermarkPositions(prev => {
      const newPositions = { ...prev };
      delete newPositions[watermarkId];
      return newPositions;
    });
  };

  // Function to unlock watermark position
  const unlockWatermark = (watermarkId) => {
    setLockedWatermarks(prev => {
      const newSet = new Set(prev);
      newSet.delete(watermarkId);
      return newSet;
    });
    
    // Remove from locked positions
    setLockedWatermarkPositions(prev => {
      const newPositions = { ...prev };
      delete newPositions[watermarkId];
      return newPositions;
    });
  };

  // Function to remove watermark completely
  const removeWatermark = (watermarkId) => {
    setLockedWatermarks(prev => {
      const newSet = new Set(prev);
      newSet.delete(watermarkId);
      return newSet;
    });
    setLocalWatermarkPositions(prev => {
      const newPositions = { ...prev };
      delete newPositions[watermarkId];
      return newPositions;
    });
    setLockedWatermarkPositions(prev => {
      const newPositions = { ...prev };
      delete newPositions[watermarkId];
      return newPositions;
    });
    onRemoveWatermark(watermarkId);
  };

  return (
    <ViewerContainer ref={containerRef} isFullscreen={isFullscreen}>
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
            onClick={() => setInternalFullscreen(!internalFullscreen)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {internalFullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
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
        
        <PDFWrapper 
          ref={pdfRef}
          scale={scale}
          dragMode={isDragMode}
        >
          {pdfUrl ? (
            <PDFPreview 
              src={pdfUrl}
              title="PDF Preview"
              onError={() => {
                console.log('PDF failed to load');
              }}
            />
          ) : (
            <PDFPlaceholder>
              <FileText size={60} color="#cbd5e0" />
              <h3>🎯 Smooth Interactive PDF Studio</h3>
              <p>Upload a PDF to start positioning watermarks</p>
              <p style={{ fontSize: '0.9rem', marginTop: '10px', color: '#9ca3af' }}>
                Optimized for buttery smooth dragging experience
              </p>
            </PDFPlaceholder>
          )}
          
          {/* Unlocked watermarks overlay */}
          <WatermarkOverlay>
            <AnimatePresence>
              {watermarks.filter(watermark => !lockedWatermarks.has(watermark.id)).map((watermark) => {
                const position = getWatermarkPosition(watermark);
                const isSelected = selectedWatermark === watermark.id;
                const isDragging = draggedWatermark === watermark.id;
                
                return (
                  <InteractiveWatermark
                    key={watermark.id}
                    style={{
                      left: position.x,
                      top: position.y,
                      transform: 'translate(-50%, -50%)',
                      zIndex: isDragging ? 1000 : (isSelected ? 100 : 20)
                    }}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ 
                      scale: isDragging ? 1.1 : 1, 
                      opacity: 1,
                      transition: { type: "spring", stiffness: 300, damping: 30 }
                    }}
                    exit={{ scale: 0, opacity: 0 }}
                    onMouseDown={(e) => handleWatermarkMouseDown(e, watermark.id)}
                    onClick={() => onSelectWatermark(watermark.id)}
                    whileHover={{ scale: isDragging ? 1.1 : 1.05 }}
                  >
                    <WatermarkContent
                      fontSize={watermark.font_size}
                      color={watermark.color}
                      opacity={watermark.opacity}
                      rotation={watermark.rotation}
                      isSelected={isSelected}
                      isLocked={false}
                      style={{
                        boxShadow: isDragging 
                          ? '0 12px 30px rgba(102, 126, 234, 0.4)' 
                          : isSelected 
                            ? '0 8px 25px rgba(102, 126, 234, 0.3)' 
                            : '0 4px 12px rgba(0,0,0,0.15)'
                      }}
                    >
                      {watermark.text}
                    </WatermarkContent>
                    
                    <AnimatePresence>
                      {isSelected && !isDragging && (
                        <WatermarkControls
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: 10 }}
                        >
                          <WatermarkControlButton
                            onClick={(e) => {
                              e.stopPropagation();
                              lockWatermark(watermark.id);
                            }}
                            title="Lock position"
                            style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#3b82f6' }}
                          >
                            <Check size={12} />
                          </WatermarkControlButton>
                          
                          <WatermarkControlButton
                            onClick={(e) => {
                              e.stopPropagation();
                              onUpdateWatermark(watermark.id, { 
                                rotation: (watermark.rotation + 15) % 360 
                              });
                            }}
                            title="Rotate watermark"
                          >
                            <RotateCw size={12} />
                          </WatermarkControlButton>
                          
                          <WatermarkControlButton
                            onClick={(e) => {
                              e.stopPropagation();
                              removeWatermark(watermark.id);
                            }}
                            title="Remove watermark"
                            style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}
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

          {/* Locked watermarks overlay - positioned relative to PDF content */}
          <LockedWatermarkOverlay scale={scale}>
            <AnimatePresence>
              {watermarks.filter(watermark => lockedWatermarks.has(watermark.id)).map((watermark) => {
                const position = getLockedWatermarkPosition(watermark);
                const isSelected = selectedWatermark === watermark.id;
                
                return (
                  <InteractiveWatermark
                    key={watermark.id}
                    style={{
                      left: position.x,
                      top: position.y,
                      transform: 'translate(-50%, -50%)',
                      zIndex: isSelected ? 100 : 20
                    }}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ 
                      scale: 1, 
                      opacity: 1,
                      transition: { type: "spring", stiffness: 300, damping: 30 }
                    }}
                    exit={{ scale: 0, opacity: 0 }}
                    onClick={() => onSelectWatermark(watermark.id)}
                    whileHover={{ scale: 1.05 }}
                  >
                    <WatermarkContent
                      fontSize={watermark.font_size}
                      color={watermark.color}
                      opacity={watermark.opacity}
                      rotation={watermark.rotation}
                      isSelected={isSelected}
                      isLocked={true}
                      style={{
                        boxShadow: isSelected 
                          ? '0 8px 25px rgba(34, 197, 94, 0.4)' 
                          : '0 4px 12px rgba(34, 197, 94, 0.3)',
                        border: '2px solid #22c55e'
                      }}
                    >
                      {watermark.text}
                      <span style={{ 
                        position: 'absolute', 
                        top: '-8px', 
                        right: '-8px', 
                        background: '#22c55e', 
                        color: 'white', 
                        borderRadius: '50%', 
                        width: '16px', 
                        height: '16px', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        fontSize: '10px',
                        fontWeight: 'bold'
                      }}>
                        ✓
                      </span>
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
                              unlockWatermark(watermark.id);
                            }}
                            title="Unlock position"
                            style={{ background: 'rgba(34, 197, 94, 0.2)', color: '#22c55e' }}
                          >
                            <RotateCw size={12} />
                          </WatermarkControlButton>
                          
                          <WatermarkControlButton
                            onClick={(e) => {
                              e.stopPropagation();
                              removeWatermark(watermark.id);
                            }}
                            title="Remove watermark"
                            style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}
                          >
                            <X size={12} />
                          </WatermarkControlButton>
                        </WatermarkControls>
                      )}
                    </AnimatePresence>
                  </InteractiveWatermark>
                );
              })}
            </AnimatePresence>
          </LockedWatermarkOverlay>
        </PDFWrapper>
      </PDFContainer>

      <StatusBar>
        <div>
          🎯 Smooth PDF Studio | {watermarks.length} watermarks
          {lockedWatermarks.size > 0 && (
            <span style={{ color: '#22c55e', marginLeft: '10px' }}>
              | {lockedWatermarks.size} locked
            </span>
          )}
        </div>
        <div>
          {draggedWatermark ? '🎯 Dragging smoothly...' : 
           lockedWatermarks.size > 0 ? 
           'Click ✓ to lock position, ✗ to remove, or 🔄 to unlock' :
           'Click and drag watermarks to position them, then click ✓ to lock'}
        </div>
      </StatusBar>
    </ViewerContainer>
  );
}

export default SmoothPDFViewer;