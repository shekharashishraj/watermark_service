import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import { X, Eye, Save, RotateCcw, Move, Target, Sparkles, Zap, Maximize, Minimize, Maximize2 } from 'lucide-react';
import ModernPDFViewer from './ModernPDFViewer';

const PositioningContainer = styled.div`
  position: ${props => props.isFullscreen ? 'fixed' : (props.isIntegrated ? 'relative' : 'relative')};
  top: ${props => props.isFullscreen ? '0' : 'auto'};
  left: ${props => props.isFullscreen ? '0' : 'auto'};
  width: ${props => props.isFullscreen ? '100vw' : (props.isIntegrated ? '100%' : 'auto')};
  height: ${props => props.isFullscreen ? '100vh' : (props.isIntegrated ? '100%' : 'auto')};
  z-index: ${props => props.isFullscreen ? '9999' : '1'};
  background: ${props => props.isFullscreen ? '#f7fafc' : (props.isIntegrated ? '#f8fafc' : 'transparent')};
  overflow: ${props => props.isFullscreen ? 'auto' : (props.isIntegrated ? 'hidden' : 'visible')};
  padding: ${props => props.isFullscreen ? '20px' : (props.isIntegrated ? '0' : '0')};
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.isFullscreen ? '15px' : '30px'};
  padding: ${props => props.isFullscreen ? '10px 20px' : '0'};
  background: ${props => props.isFullscreen ? 'rgba(255, 255, 255, 0.95)' : 'transparent'};
  backdrop-filter: ${props => props.isFullscreen ? 'blur(10px)' : 'none'};
  border-radius: ${props => props.isFullscreen ? '15px' : '0'};
  box-shadow: ${props => props.isFullscreen ? '0 4px 20px rgba(0,0,0,0.1)' : 'none'};
  position: ${props => props.isFullscreen ? 'sticky' : 'static'};
  top: ${props => props.isFullscreen ? '20px' : 'auto'};
  z-index: 10;
`;

const Title = styled.h2`
  font-size: ${props => props.isFullscreen ? '2.2rem' : '1.8rem'};
  font-weight: 600;
  color: #1a202c;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0;
  
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const HeaderControls = styled.div`
  display: flex;
  gap: 10px;
  align-items: center;
`;

const HeaderButton = styled.button`
  background: ${props => props.variant === 'close' ? '#fed7d7' : props.variant === 'fullscreen' ? '#e6fffa' : '#f7fafc'};
  color: ${props => props.variant === 'close' ? '#c53030' : props.variant === 'fullscreen' ? '#00695c' : '#4a5568'};
  border: none;
  border-radius: 8px;
  padding: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  font-size: 0.9rem;
  gap: 6px;

  &:hover {
    background: ${props => props.variant === 'close' ? '#feb2b2' : props.variant === 'fullscreen' ? '#b2dfdb' : '#edf2f7'};
    transform: scale(1.05);
  }
`;

const Instructions = styled.div`
  background: #ebf8ff;
  border: 1px solid #bee3f8;
  border-radius: 12px;
  padding: ${props => props.isFullscreen ? '15px 20px' : '20px'};
  margin-bottom: ${props => props.isFullscreen ? '15px' : '30px'};
  display: ${props => props.isFullscreen && props.isCollapsed ? 'none' : 'block'};
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


const Controls = styled.div`
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 20px;
  padding: ${props => props.isFullscreen ? '15px' : '0'};
  background: ${props => props.isFullscreen ? 'rgba(255, 255, 255, 0.95)' : 'transparent'};
  backdrop-filter: ${props => props.isFullscreen ? 'blur(10px)' : 'none'};
  border-radius: ${props => props.isFullscreen ? '15px' : '0'};
  box-shadow: ${props => props.isFullscreen ? '0 -4px 20px rgba(0,0,0,0.1)' : 'none'};
  position: ${props => props.isFullscreen ? 'sticky' : 'static'};
  bottom: ${props => props.isFullscreen ? '20px' : 'auto'};
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

function VisualPositioning({ watermarks, onUpdateProperty, onRemoveWatermark, onClose, currentFileId, apiBaseUrl, pdfInfo, isIntegrated = false }) {
  const [selectedWatermark, setSelectedWatermark] = useState(null);
  const [localWatermarks, setLocalWatermarks] = useState(watermarks);
  const [hasChanges, setHasChanges] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isInstructionsCollapsed, setIsInstructionsCollapsed] = useState(isIntegrated);

  const handleUpdateWatermark = (watermarkId, updates) => {
    // Update local state immediately
    setLocalWatermarks(prev => 
      prev.map(w => w.id === watermarkId ? { ...w, ...updates } : w)
    );
    setHasChanges(true);
    
    // Send updates to parent
    Object.entries(updates).forEach(([key, value]) => {
      onUpdateProperty(watermarkId, key, value);
    });
  };

  const handleSelectWatermark = (watermarkId) => {
    setSelectedWatermark(watermarkId);
  };

  const resetPositions = () => {
    const updatedWatermarks = localWatermarks.map(w => ({
      ...w,
      position: 'center',
      custom_x: '',
      custom_y: ''
    }));
    
    setLocalWatermarks(updatedWatermarks);
    setHasChanges(true);
    
    // Notify parent
    localWatermarks.forEach(watermark => {
      onUpdateProperty(watermark.id, 'position', 'center');
      onUpdateProperty(watermark.id, 'custom_x', '');
      onUpdateProperty(watermark.id, 'custom_y', '');
    });
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    if (!isFullscreen) {
      setIsInstructionsCollapsed(true);
    }
  };

  const handleClose = () => {
    if (isFullscreen) {
      setIsFullscreen(false);
    } else {
      onClose();
    }
  };

  // Handle ESC key to exit fullscreen
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };

    if (isFullscreen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'auto';
    };
  }, [isFullscreen]);

  // Sync local watermarks with props
  useEffect(() => {
    const uniqueWatermarks = watermarks.filter((watermark, index, self) => 
      index === self.findIndex(w => w.id === watermark.id)
    );
    setLocalWatermarks(uniqueWatermarks);
  }, [watermarks]);

  return (
    <AnimatePresence>
      <PositioningContainer 
        isFullscreen={isFullscreen}
        isIntegrated={isIntegrated}
        initial={isFullscreen ? { opacity: 0, scale: 0.95 } : false}
        animate={isFullscreen ? { opacity: 1, scale: 1 } : false}
        exit={isFullscreen ? { opacity: 0, scale: 0.95 } : false}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        as={motion.div}
      >
        {!isIntegrated && (
          <Header isFullscreen={isFullscreen}>
            <Title isFullscreen={isFullscreen}>
              <Sparkles size={isFullscreen ? 28 : 24} />
              {isFullscreen ? '🚀 Fullscreen Watermarking Studio' : '🎯 Smooth PDF Watermarking Studio'}
            </Title>
            <HeaderControls>
              {!isFullscreen && (
                <HeaderButton 
                  variant="fullscreen" 
                  onClick={toggleFullscreen}
                  title="Enter Fullscreen Mode"
                >
                  <Maximize2 size={18} />
                  Fullscreen
                </HeaderButton>
              )}
              {isFullscreen && (
                <HeaderButton 
                  variant="minimize" 
                  onClick={() => setIsInstructionsCollapsed(!isInstructionsCollapsed)}
                  title="Toggle Instructions"
                >
                  <Eye size={18} />
                  {isInstructionsCollapsed ? 'Show Help' : 'Hide Help'}
                </HeaderButton>
              )}
              <HeaderButton 
                variant="close" 
                onClick={handleClose}
                title={isFullscreen ? "Exit Fullscreen (ESC)" : "Close Studio"}
              >
                {isFullscreen ? <Minimize size={18} /> : <X size={18} />}
                {isFullscreen ? 'Exit' : 'Close'}
              </HeaderButton>
            </HeaderControls>
          </Header>
        )}

        {!isIntegrated && (
          <Instructions isFullscreen={isFullscreen} isCollapsed={isInstructionsCollapsed}>
            <InstructionsTitle>
              <Zap size={20} />
              {isFullscreen ? '⚡ Pro Watermarking Tools' : '🚀 Optimized Performance Controls'}
            </InstructionsTitle>
            <InstructionsList>
              <InstructionItem>
                <Move size={16} />
                {isFullscreen ? 'Full-screen precision dragging' : 'Butter-smooth dragging with intelligent throttling'}
              </InstructionItem>
              <InstructionItem>
                <Target size={16} />
                {isFullscreen ? 'Maximum workspace for accurate positioning' : 'Precise positioning with zoom controls and grid overlay'}
              </InstructionItem>
              <InstructionItem>
                <Eye size={16} />
                {isFullscreen ? 'Enhanced visibility for detailed work' : 'Click watermarks to select and access lock/rotation controls'}
              </InstructionItem>
              <InstructionItem>
                <Save size={16} />
                {isFullscreen ? 'Press ESC to exit fullscreen mode' : 'Click ✓ to lock positions, ✗ to remove watermarks'}
              </InstructionItem>
            </InstructionsList>
          </Instructions>
        )}

        <ModernPDFViewer
          currentFileId={currentFileId}
          apiBaseUrl={apiBaseUrl}
          watermarks={localWatermarks}
          onUpdateWatermark={handleUpdateWatermark}
          onRemoveWatermark={onRemoveWatermark}
          selectedWatermark={selectedWatermark}
          onSelectWatermark={handleSelectWatermark}
          isFullscreenMode={isFullscreen}
          isIntegrated={isIntegrated}
          pdfInfo={pdfInfo}
        />

        {!isIntegrated && (
          <Controls isFullscreen={isFullscreen}>
            <Button
              className="secondary"
              onClick={resetPositions}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <RotateCcw size={16} />
              Reset All Positions
            </Button>
            {!isFullscreen && (
              <Button
                className="secondary"
                onClick={toggleFullscreen}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Maximize size={16} />
                Fullscreen Mode
              </Button>
            )}
            <Button
              className="primary"
              onClick={isFullscreen ? () => setIsFullscreen(false) : onClose}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Save size={16} />
              {hasChanges ? 'Save & Close' : (isFullscreen ? 'Exit Fullscreen' : 'Close Studio')}
            </Button>
          </Controls>
        )}
      </PositioningContainer>
    </AnimatePresence>
  );
}

export default VisualPositioning;
