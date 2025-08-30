import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import { X, Lock, Unlock, RotateCcw, Move, ZoomIn, ZoomOut } from 'lucide-react';

const ViewerContainer = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
  overflow: hidden;
`;

const ViewerToolbar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: white;
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
`;

const ToolbarSection = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ZoomControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  background: #f7fafc;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
`;

const ZoomButton = styled.button`
  background: none;
  border: none;
  padding: 4px;
  cursor: pointer;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s ease;
  
  &:hover {
    background: #edf2f7;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ZoomLevel = styled.span`
  font-size: 0.8rem;
  font-weight: 600;
  color: #4a5568;
  min-width: 40px;
  text-align: center;
`;

const PDFContainer = styled.div`
  flex: 1;
  position: relative;
  overflow: auto;
  background: #f1f5f9;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 20px;
`;

const PDFWrapper = styled.div`
  position: relative;
  background: white;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
  border-radius: 8px;
  overflow: hidden;
  transform: scale(${props => props.zoom});
  transform-origin: top center;
  transition: transform 0.2s ease;
`;

const PDFFrame = styled.iframe`
  width: ${props => props.width || 800}px;
  height: ${props => props.height || 1000}px;
  border: none;
  display: block;
  pointer-events: ${props => props.interactionMode ? 'none' : 'auto'};
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
  cursor: ${props => props.isDragging ? 'grabbing' : (props.isLocked ? 'default' : 'grab')};
  user-select: none;
  pointer-events: ${props => props.isLocked ? 'none' : 'auto'};
  transform-origin: center;
  z-index: ${props => props.isSelected ? '20' : '15'};
  
  &:hover {
    z-index: 20;
  }
`;

const WatermarkText = styled.div`
  font-family: Arial, sans-serif;
  font-size: ${props => props.fontSize}px;
  color: ${props => props.color};
  opacity: ${props => props.opacity};
  transform: rotate(${props => props.rotation}deg);
  white-space: pre-wrap;
  text-shadow: ${props => props.isSelected ? '0 0 4px rgba(102, 126, 234, 0.5)' : 'none'};
  transition: text-shadow 0.2s ease;
  border: ${props => props.isSelected ? '2px dashed #667eea' : 'none'};
  padding: ${props => props.isSelected ? '4px 8px' : '0'};
  border-radius: 4px;
  line-height: 1.2;
  text-align: center;
`;

const WatermarkControls = styled(motion.div)`
  position: absolute;
  top: -35px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 4px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 8px;
  padding: 6px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  border: 1px solid #e2e8f0;
`;

const ControlButton = styled.button`
  background: ${props => props.variant === 'lock' ? '#e6fffa' : props.variant === 'remove' ? '#fed7d7' : '#f7fafc'};
  color: ${props => props.variant === 'lock' ? '#00695c' : props.variant === 'remove' ? '#c53030' : '#4a5568'};
  border: none;
  border-radius: 4px;
  padding: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  font-size: 12px;
  
  &:hover {
    background: ${props => props.variant === 'lock' ? '#b2dfdb' : props.variant === 'remove' ? '#feb2b2' : '#edf2f7'};
    transform: scale(1.05);
  }
`;

const StatusInfo = styled.div`
  font-size: 0.8rem;
  color: #718096;
  display: flex;
  align-items: center;
  gap: 6px;
`;

function ModernPDFViewer({ 
  currentFileId, 
  apiBaseUrl, 
  watermarks = [], 
  onUpdateWatermark,
  onRemoveWatermark,
  selectedWatermark,
  onSelectWatermark,
  isFullscreenMode = false,
  isIntegrated = false,
  pdfInfo
}) {
  const [zoom, setZoom] = useState(0.8);
  const [isDragging, setIsDragging] = useState(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [localWatermarkPositions, setLocalWatermarkPositions] = useState({});
  const [lockedWatermarks, setLockedWatermarks] = useState(new Set());
  const [pdfDimensions, setPdfDimensions] = useState({ width: 800, height: 1000 });
  const containerRef = useRef(null);
  
  // Update PDF dimensions when pdfInfo changes
  useEffect(() => {
    if (pdfInfo && pdfInfo.page_size) {
      setPdfDimensions({
        width: pdfInfo.page_size.width,
        height: pdfInfo.page_size.height
      });
    }
  }, [pdfInfo]);
  
  // Text processing functions (similar to backend logic)
  const wrapText = (text, maxWidth, fontSize) => {
    const words = text.split(' ');
    const lines = [];
    let currentLine = [];
    
    for (const word of words) {
      const testLine = currentLine.concat([word]).join(' ');
      const testWidth = testLine.length * fontSize * 0.6; // Approximate character width
      
      if (testWidth <= maxWidth) {
        currentLine.push(word);
      } else {
        if (currentLine.length > 0) {
          lines.push(currentLine.join(' '));
          currentLine = [word];
        } else {
          // Single word is too long, break it
          lines.push(word);
        }
      }
    }
    
    if (currentLine.length > 0) {
      lines.push(currentLine.join(' '));
    }
    
    return lines;
  };
  
  const calculateOptimalFontSize = (text, maxWidth, maxHeight, initialFontSize) => {
    let fontSize = initialFontSize;
    
    while (fontSize > 8) {
      const textWidth = text.length * fontSize * 0.6; // Approximate character width
      
      if (textWidth <= maxWidth) {
        if (fontSize <= maxHeight) {
          return { fontSize, lines: [text] };
        } else {
          fontSize -= 1;
        }
      } else {
        // Text is too wide, try wrapping
        const wrappedLines = wrapText(text, maxWidth, fontSize);
        const totalHeight = wrappedLines.length * fontSize;
        
        if (totalHeight <= maxHeight) {
          return { fontSize, lines: wrappedLines };
        } else {
          fontSize -= 1;
        }
      }
    }
    
    // If we get here, use minimum font size and wrap
    const wrappedLines = wrapText(text, maxWidth, 8);
    return { fontSize: 8, lines: wrappedLines };
  };
  
  // Throttle updates to improve performance
  const throttledUpdate = useCallback(
    (() => {
      let timeout = null;
      return (watermarkId, updates) => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => {
          onUpdateWatermark(watermarkId, updates);
        }, 100);
      };
    })(),
    [onUpdateWatermark]
  );

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.1, 2.0));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.1, 0.3));
  };

  const resetZoom = () => {
    setZoom(0.8);
  };

  const getWatermarkPosition = (watermark) => {
    const localPos = localWatermarkPositions[watermark.id];
    if (localPos) {
      return localPos;
    }

    if (watermark.position === 'custom' && watermark.custom_x && watermark.custom_y) {
      return { x: parseInt(watermark.custom_x), y: parseInt(watermark.custom_y) };
    }

    const centerX = pdfDimensions.width / 2;
    const centerY = pdfDimensions.height / 2;
    
    switch (watermark.position) {
      case 'top-left': return { x: 50, y: 50 };
      case 'top-center': return { x: centerX, y: 50 };
      case 'top-right': return { x: pdfDimensions.width - 50, y: 50 };
      case 'center-left': return { x: 50, y: centerY };
      case 'center': return { x: centerX, y: centerY };
      case 'center-right': return { x: pdfDimensions.width - 50, y: centerY };
      case 'bottom-left': return { x: 50, y: pdfDimensions.height - 50 };
      case 'bottom-center': return { x: centerX, y: pdfDimensions.height - 50 };
      case 'bottom-right': return { x: pdfDimensions.width - 50, y: pdfDimensions.height - 50 };
      default: return { x: centerX, y: centerY };
    }
  };

  const constrainPosition = (x, y, element) => {
    if (!element) return { x, y };
    
    const rect = element.getBoundingClientRect();
    const padding = 20;
    
    const minX = padding;
    const minY = padding;
    const maxX = pdfDimensions.width - rect.width - padding;
    const maxY = pdfDimensions.height - rect.height - padding;
    
    return {
      x: Math.max(minX, Math.min(maxX, x)),
      y: Math.max(minY, Math.min(maxY, y))
    };
  };
  
  const getWatermarkDimensions = (watermark) => {
    const margin = 20;
    const availableWidth = pdfDimensions.width - 2 * margin;
    const availableHeight = pdfDimensions.height - 2 * margin;
    const textProcessing = calculateOptimalFontSize(
      watermark.text, 
      availableWidth, 
      availableHeight, 
      watermark.font_size
    );
    
    // Calculate approximate dimensions
    const maxLineWidth = Math.max(...textProcessing.lines.map(line => line.length * textProcessing.fontSize * 0.6));
    const totalHeight = textProcessing.lines.length * textProcessing.fontSize * 1.2;
    
    return {
      width: maxLineWidth,
      height: totalHeight
    };
  };

  const handleMouseDown = (e, watermarkId) => {
    if (lockedWatermarks.has(watermarkId)) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();
    
    if (containerRect) {
      setDragOffset({
        x: (e.clientX - rect.left) / zoom,
        y: (e.clientY - rect.top) / zoom
      });
    }
    
    setIsDragging(watermarkId);
    onSelectWatermark(watermarkId);
  };

  const handleMouseMove = useCallback((e) => {
    if (!isDragging || !containerRef.current) return;
    
    const containerRect = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - containerRect.left - dragOffset.x * zoom) / zoom;
    const y = (e.clientY - containerRect.top - dragOffset.y * zoom) / zoom;
    
    const watermarkElement = document.getElementById(`watermark-${isDragging}`);
    const constrainedPos = constrainPosition(x, y, watermarkElement);
    
    setLocalWatermarkPositions(prev => ({
      ...prev,
      [isDragging]: constrainedPos
    }));
    
    throttledUpdate(isDragging, {
      position: 'custom',
      custom_x: Math.round(constrainedPos.x).toString(),
      custom_y: Math.round(constrainedPos.y).toString()
    });
  }, [isDragging, dragOffset, zoom, throttledUpdate]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(null);
    setDragOffset({ x: 0, y: 0 });
  }, []);

  const toggleLock = (watermarkId) => {
    setLockedWatermarks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(watermarkId)) {
        newSet.delete(watermarkId);
      } else {
        newSet.add(watermarkId);
      }
      return newSet;
    });
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const pdfUrl = currentFileId ? `${apiBaseUrl}/original/${currentFileId}` : null;

  return (
    <ViewerContainer>
      <ViewerToolbar>
        <ToolbarSection>
          <StatusInfo>
            <Move size={14} />
            {watermarks.length} watermark{watermarks.length !== 1 ? 's' : ''} • {lockedWatermarks.size} locked
          </StatusInfo>
        </ToolbarSection>
        
        <ToolbarSection>
          <ZoomControls>
            <ZoomButton onClick={handleZoomOut} disabled={zoom <= 0.3}>
              <ZoomOut size={14} />
            </ZoomButton>
            <ZoomLevel>{Math.round(zoom * 100)}%</ZoomLevel>
            <ZoomButton onClick={handleZoomIn} disabled={zoom >= 2.0}>
              <ZoomIn size={14} />
            </ZoomButton>
          </ZoomControls>
          
          <ZoomButton onClick={resetZoom} title="Reset Zoom">
            <RotateCcw size={14} />
          </ZoomButton>
        </ToolbarSection>
      </ViewerToolbar>
      
      <PDFContainer ref={containerRef}>
        {pdfUrl && (
          <PDFWrapper zoom={zoom}>
            <PDFFrame 
              src={pdfUrl} 
              interactionMode={watermarks.length > 0}
              title="PDF Preview"
              width={pdfDimensions.width}
              height={pdfDimensions.height}
            />
            
            <WatermarkOverlay>
              <AnimatePresence>
                {watermarks.map((watermark) => {
                  const position = getWatermarkPosition(watermark);
                  const isSelected = selectedWatermark === watermark.id;
                  const isLocked = lockedWatermarks.has(watermark.id);
                  const isDraggingThis = isDragging === watermark.id;
                  
                  // Calculate optimal text display (similar to backend logic)
                  const margin = 20;
                  const availableWidth = pdfDimensions.width - 2 * margin;
                  const availableHeight = pdfDimensions.height - 2 * margin;
                  const textProcessing = calculateOptimalFontSize(
                    watermark.text, 
                    availableWidth, 
                    availableHeight, 
                    watermark.font_size
                  );
                  
                  return (
                    <InteractiveWatermark
                      key={watermark.id}
                      id={`watermark-${watermark.id}`}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ 
                        opacity: 1, 
                        scale: 1,
                        x: position.x,
                        y: position.y
                      }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ type: "spring", stiffness: 300, damping: 25 }}
                      isSelected={isSelected}
                      isLocked={isLocked}
                      isDragging={isDraggingThis}
                      onMouseDown={(e) => handleMouseDown(e, watermark.id)}
                      onClick={() => onSelectWatermark(watermark.id)}
                    >
                      <WatermarkText
                        fontSize={textProcessing.fontSize}
                        color={watermark.color}
                        opacity={watermark.opacity}
                        rotation={watermark.rotation}
                        isSelected={isSelected}
                        title={`Text will be automatically adjusted to fit within page boundaries. Adjusted size: ${textProcessing.fontSize}px (original: ${watermark.font_size}px)`}
                      >
                        {textProcessing.lines.join('\n')}
                      </WatermarkText>
                      
                      {isSelected && !isDraggingThis && (
                        <WatermarkControls
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: 10 }}
                        >
                          <ControlButton 
                            variant="lock"
                            onClick={() => toggleLock(watermark.id)}
                            title={isLocked ? 'Unlock watermark' : 'Lock watermark'}
                          >
                            {isLocked ? <Unlock size={12} /> : <Lock size={12} />}
                          </ControlButton>
                          
                          <ControlButton 
                            variant="remove"
                            onClick={() => onRemoveWatermark(watermark.id)}
                            title="Remove watermark"
                          >
                            <X size={12} />
                          </ControlButton>
                        </WatermarkControls>
                      )}
                    </InteractiveWatermark>
                  );
                })}
              </AnimatePresence>
            </WatermarkOverlay>
          </PDFWrapper>
        )}
      </PDFContainer>
    </ViewerContainer>
  );
}

export default ModernPDFViewer;