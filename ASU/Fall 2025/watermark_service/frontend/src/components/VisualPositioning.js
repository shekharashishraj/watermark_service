import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { X, Eye, Save, RotateCcw, Move, Target, FileText } from 'lucide-react';

const PositioningContainer = styled.div`
  position: relative;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
`;

const Title = styled.h2`
  font-size: 1.8rem;
  font-weight: 600;
  color: #1a202c;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const CloseButton = styled.button`
  background: #fed7d7;
  color: #c53030;
  border: none;
  border-radius: 8px;
  padding: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;

  &:hover {
    background: #feb2b2;
    transform: scale(1.1);
  }
`;

const Instructions = styled.div`
  background: #ebf8ff;
  border: 1px solid #bee3f8;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 30px;
`;

const InstructionsTitle = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  color: #2b6cb0;
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const InstructionsList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
`;

const InstructionItem = styled.li`
  color: #4a5568;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;

  &:before {
    content: "•";
    color: #667eea;
    font-weight: bold;
  }
`;

const PreviewContainer = styled.div`
  position: relative;
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  min-height: 600px;
  overflow: hidden;
`;

const PDFPreview = styled.div`
  position: relative;
  width: 100%;
  height: 560px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
`;

const PDFContent = styled.div`
  width: 100%;
  height: 100%;
  background: 
    linear-gradient(45deg, #f0f0f0 25%, transparent 25%), 
    linear-gradient(-45deg, #f0f0f0 25%, transparent 25%), 
    linear-gradient(45deg, transparent 75%, #f0f0f0 75%), 
    linear-gradient(-45deg, transparent 75%, #f0f0f0 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  position: relative;
  overflow: hidden;
`;

const PDFBackground = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: white;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
`;

const PDFImage = styled.object`
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  border: 1px solid #e2e8f0;
  border-radius: 4px;
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
  padding: 20px;
`;

const PDFIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 15px;
  color: #cbd5e0;
`;

const DraggableWatermark = styled(motion.div)`
  position: absolute;
  cursor: move;
  user-select: none;
  padding: 8px 12px;
  border-radius: 6px;
  background: rgba(102, 126, 234, 0.9);
  color: white;
  font-weight: 600;
  font-size: ${props => props.fontSize}px;
  color: ${props => props.color};
  opacity: ${props => props.opacity};
  transform: rotate(${props => props.rotation}deg);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  border: 2px solid rgba(255,255,255,0.3);
  z-index: 20;
  transition: all 0.3s ease;
  backdrop-filter: blur(2px);

  &:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    transform: rotate(${props => props.rotation}deg) scale(1.05);
  }

  &.dragging {
    z-index: 100;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
  }
`;

const WatermarkHandle = styled.div`
  position: absolute;
  top: -8px;
  right: -8px;
  width: 20px;
  height: 20px;
  background: #c53030;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: white;
  font-size: 12px;
  font-weight: bold;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  transition: all 0.3s ease;

  &:hover {
    background: #e53e3e;
    transform: scale(1.2);
  }
`;

const PositionIndicator = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0,0,0,0.8);
  color: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  z-index: 50;
  pointer-events: none;
`;

const Controls = styled.div`
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 20px;
`;

const Button = styled(motion.button)`
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;

  &.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
  }

  &.secondary {
    background: #f7fafc;
    color: #4a5568;
    border: 2px solid #e2e8f0;
    
    &:hover {
      background: #edf2f7;
      border-color: #cbd5e0;
    }
  }
`;

function VisualPositioning({ watermarks, onUpdateProperty, onRemoveWatermark, onClose, currentFileId, apiBaseUrl }) {
  const [isDragging, setIsDragging] = useState(false);
  const [draggedElement, setDraggedElement] = useState(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [positionIndicator, setPositionIndicator] = useState({ x: 0, y: 0, visible: false });
  const [pdfUrl, setPdfUrl] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [localWatermarks, setLocalWatermarks] = useState(watermarks);
  const previewRef = useRef(null);

  const handleMouseDown = (e, watermarkId) => {
    e.preventDefault();
    setIsDragging(true);
    setDraggedElement(watermarkId);
    
    const rect = e.currentTarget.getBoundingClientRect();
    const previewRect = previewRef.current.getBoundingClientRect();
    
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !draggedElement) return;

    const previewRect = previewRef.current.getBoundingClientRect();
    const x = e.clientX - previewRect.left - dragOffset.x;
    const y = e.clientY - previewRect.top - dragOffset.y;

    // Constrain to preview bounds
    const constrainedX = Math.max(0, Math.min(x, previewRect.width - 100));
    const constrainedY = Math.max(0, Math.min(y, previewRect.height - 50));

    // Update position indicator
    setPositionIndicator({
      x: constrainedX,
      y: constrainedY,
      visible: true
    });

    // Convert to PDF coordinates (assuming A4 ratio)
    const pdfX = (constrainedX / previewRect.width) * 595; // A4 width in points
    const pdfY = (constrainedY / previewRect.height) * 842; // A4 height in points
    
    console.log(`Dragging watermark ${draggedElement} to PDF coordinates: x=${Math.round(pdfX)}, y=${Math.round(pdfY)}`); // Debug

    // Update local state immediately for real-time feedback
    setLocalWatermarks(prev => prev.map(w => 
      w.id === draggedElement 
        ? { 
            ...w, 
            position: 'custom',
            custom_x: Math.round(pdfX),
            custom_y: Math.round(pdfY)
          }
        : w
    ));
  };

  const handleMouseUp = () => {
    if (draggedElement) {
      // Send final position to backend
      const watermark = localWatermarks.find(w => w.id === draggedElement);
      if (watermark && watermark.position === 'custom') {
        onUpdateProperty(draggedElement, 'position', 'custom');
        onUpdateProperty(draggedElement, 'custom_x', watermark.custom_x);
        onUpdateProperty(draggedElement, 'custom_y', watermark.custom_y);
      }
    }
    
    setIsDragging(false);
    setDraggedElement(null);
    setPositionIndicator(prev => ({ ...prev, visible: false }));
  };

  useEffect(() => {
    if (isDragging) {
      const handleMouseMoveEvent = (e) => handleMouseMove(e);
      const handleMouseUpEvent = () => handleMouseUp();
      
      document.addEventListener('mousemove', handleMouseMoveEvent);
      document.addEventListener('mouseup', handleMouseUpEvent);
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMoveEvent);
        document.removeEventListener('mouseup', handleMouseUpEvent);
      };
    }
  }, [isDragging, draggedElement, dragOffset]);

  const getWatermarkPosition = (watermark) => {
    const previewRect = previewRef.current?.getBoundingClientRect();
    if (!previewRect) {
      return { x: 250, y: 250 }; // Default center position
    }

    if (watermark.position === 'custom' && watermark.custom_x !== undefined && watermark.custom_y !== undefined) {
      // Convert PDF coordinates back to screen coordinates
      const x = (watermark.custom_x / 595) * previewRect.width;
      const y = (watermark.custom_y / 842) * previewRect.height;
      console.log(`Watermark ${watermark.id} custom position: PDF(${watermark.custom_x}, ${watermark.custom_y}) -> Screen(${x}, ${y})`);
      return { x, y };
    }
    
    // Default positions based on preview size
    const centerX = previewRect.width / 2;
    const centerY = previewRect.height / 2;
    const margin = 50;
    
    const positions = {
      'top-left': { x: margin, y: margin },
      'top-center': { x: centerX, y: margin },
      'top-right': { x: previewRect.width - margin, y: margin },
      'center-left': { x: margin, y: centerY },
      'center': { x: centerX, y: centerY },
      'center-right': { x: previewRect.width - margin, y: centerY },
      'bottom-left': { x: margin, y: previewRect.height - margin },
      'bottom-center': { x: centerX, y: previewRect.height - margin },
      'bottom-right': { x: previewRect.width - margin, y: previewRect.height - margin }
    };
    
    return positions[watermark.position] || positions.center;
  };

  const resetPositions = () => {
    localWatermarks.forEach(watermark => {
      onUpdateProperty(watermark.id, 'position', 'center');
      onUpdateProperty(watermark.id, 'custom_x', '');
      onUpdateProperty(watermark.id, 'custom_y', '');
    });
    
    // Update local state
    setLocalWatermarks(prev => prev.map(w => ({
      ...w,
      position: 'center',
      custom_x: '',
      custom_y: ''
    })));
  };

  // Load PDF for visual positioning
  useEffect(() => {
    if (currentFileId && apiBaseUrl) {
      setPdfLoading(true);
      // Use the preview endpoint to get the PDF for visual positioning
      setPdfUrl(`${apiBaseUrl}/preview/${currentFileId}`);
      setPdfLoading(false);
    }
  }, [currentFileId, apiBaseUrl]);

  // Sync local watermarks with props and remove duplicates
  useEffect(() => {
    // Remove duplicates based on ID
    const uniqueWatermarks = watermarks.filter((watermark, index, self) => 
      index === self.findIndex(w => w.id === watermark.id)
    );
    setLocalWatermarks(uniqueWatermarks);
  }, [watermarks]);

  return (
    <PositioningContainer>
      <Header>
        <Title>
          <Eye size={24} />
          Visual Watermark Positioning
        </Title>
        <CloseButton onClick={onClose}>
          <X size={20} />
        </CloseButton>
      </Header>

      <Instructions>
        <InstructionsTitle>
          <Target size={20} />
          Drag and Drop Instructions
        </InstructionsTitle>
        <InstructionsList>
          <InstructionItem>
            <Move size={16} />
            Click and drag watermarks to position them anywhere on the PDF
          </InstructionItem>
          <InstructionItem>
            <Target size={16} />
            Watch the coordinates update in real-time as you drag
          </InstructionItem>
          <InstructionItem>
            <Save size={16} />
            Click 'Save Positions' to apply your changes
          </InstructionItem>
        </InstructionsList>
      </Instructions>

      <PreviewContainer>
        <PDFPreview ref={previewRef}>
          <PDFContent>
            <PDFBackground>
              {pdfLoading ? (
                <PDFPlaceholder>
                  <PDFIcon>
                    <FileText size={60} />
                  </PDFIcon>
                  <div>Loading PDF preview...</div>
                </PDFPlaceholder>
              ) : pdfUrl ? (
                <PDFImage 
                  data={pdfUrl} 
                  type="application/pdf"
                  onError={() => {
                    // Fallback to placeholder if PDF loading fails
                    setPdfUrl(null);
                  }}
                >
                  <PDFPlaceholder>
                    <PDFIcon>
                      <FileText size={60} />
                    </PDFIcon>
                    <div>PDF preview not available</div>
                    <div style={{ fontSize: '0.9rem', marginTop: '10px' }}>
                      Drag watermarks to position them on your PDF
                    </div>
                  </PDFPlaceholder>
                </PDFImage>
              ) : (
                <PDFPlaceholder>
                  <PDFIcon>
                    <FileText size={60} />
                  </PDFIcon>
                  <div>PDF preview will appear here</div>
                  <div style={{ fontSize: '0.9rem', marginTop: '10px' }}>
                    Drag watermarks to position them on your PDF
                  </div>
                </PDFPlaceholder>
              )}
            </PDFBackground>
            
            {localWatermarks.map((watermark) => {
              const position = getWatermarkPosition(watermark);
              return (
                <DraggableWatermark
                  key={watermark.id}
                  style={{
                    left: position.x,
                    top: position.y
                  }}
                  className={draggedElement === watermark.id ? 'dragging' : ''}
                  fontSize={watermark.font_size}
                  color={watermark.color}
                  opacity={watermark.opacity}
                  rotation={watermark.rotation}
                  onMouseDown={(e) => handleMouseDown(e, watermark.id)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {watermark.text}
                  <WatermarkHandle
                    onClick={(e) => {
                      e.stopPropagation();
                      // Remove watermark from local state
                      setLocalWatermarks(prev => prev.filter(w => w.id !== watermark.id));
                      // Notify parent to remove watermark
                      if (onRemoveWatermark) {
                        onRemoveWatermark(watermark.id);
                      }
                    }}
                  >
                    ×
                  </WatermarkHandle>
                </DraggableWatermark>
              );
            })}
          </PDFContent>
        </PDFPreview>

        {positionIndicator.visible && (
          <PositionIndicator>
            X: {Math.round(positionIndicator.x)}, Y: {Math.round(positionIndicator.y)}
          </PositionIndicator>
        )}
      </PreviewContainer>

      <Controls>
        <Button
          className="secondary"
          onClick={resetPositions}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw size={16} />
          Reset Positions
        </Button>
        <Button
          className="primary"
          onClick={onClose}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Save size={16} />
          Save Positions
        </Button>
      </Controls>
    </PositioningContainer>
  );
}

export default VisualPositioning;
