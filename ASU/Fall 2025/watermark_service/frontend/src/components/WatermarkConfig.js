import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { Plus, X, Eye, Settings, Type, FileText, CheckSquare } from 'lucide-react';
import { ChromePicker } from 'react-color';

const ConfigContainer = styled.div``;

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

const ActionButtons = styled.div`
  display: flex;
  gap: 15px;
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

  &.danger {
    background: #fed7d7;
    color: #c53030;
    border: 2px solid #feb2b2;
    
    &:hover {
      background: #feb2b2;
    }
  }
`;

const WatermarksList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.isCompact ? '12px' : '20px'};
  padding: ${props => props.isCompact ? '24px' : '0'};
`;

const WatermarkCard = styled(motion.div)`
  background: ${props => props.isCompact ? 'white' : '#f8fafc'};
  border: ${props => props.isCompact ? '1px solid #e2e8f0' : '2px solid #e2e8f0'};
  border-radius: ${props => props.isCompact ? '8px' : '12px'};
  padding: ${props => props.isCompact ? '16px' : '25px'};
  position: relative;

  &:hover {
    border-color: #cbd5e0;
  }
`;

const WatermarkHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.isCompact ? '12px' : '20px'};
`;

const WatermarkTitle = styled.h3`
  font-size: ${props => props.isCompact ? '1rem' : '1.2rem'};
  font-weight: 600;
  color: #1a202c;
  display: flex;
  align-items: center;
  gap: 6px;
`;

const RemoveButton = styled.button`
  background: #fed7d7;
  color: #c53030;
  border: none;
  border-radius: 6px;
  padding: 8px;
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

const ConfigGrid = styled.div`
  display: grid;
  grid-template-columns: ${props => props.isCompact ? '1fr' : 'repeat(auto-fit, minmax(250px, 1fr))'};
  gap: ${props => props.isCompact ? '12px' : '20px'};
`;

const ConfigGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const Label = styled.label`
  font-weight: 600;
  color: #4a5568;
  font-size: 0.9rem;
`;

const Input = styled.input`
  padding: 12px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.3s ease;

  &:focus {
    outline: none;
    border-color: #667eea;
  }
`;

const RangeContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const RangeInput = styled.input`
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: #e2e8f0;
  outline: none;
  -webkit-appearance: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  }

  &::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  }
`;

const RangeValue = styled.span`
  font-weight: 600;
  color: #667eea;
  min-width: 40px;
  text-align: center;
`;

const ColorContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
`;

const ColorPicker = styled.div`
  position: relative;
`;

const ColorButton = styled.button`
  width: 40px;
  height: 40px;
  border: 3px solid #e2e8f0;
  border-radius: 8px;
  background: ${props => props.color};
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    transform: scale(1.1);
  }
`;

const ColorPickerPopover = styled.div`
  position: absolute;
  top: 50px;
  left: 0;
  z-index: 1000;
`;

const ColorPickerCover = styled.div`
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
`;

const PageSelector = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const PageOptions = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const PageOption = styled.button`
  padding: 8px 12px;
  border: 2px solid ${props => props.selected ? '#667eea' : '#e2e8f0'};
  border-radius: 6px;
  background: ${props => props.selected ? '#667eea' : 'white'};
  color: ${props => props.selected ? 'white' : '#4a5568'};
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 4px;

  &:hover {
    border-color: #667eea;
    background: ${props => props.selected ? '#5a67d8' : '#f7fafc'};
  }

  &.all-pages {
    background: ${props => props.selected ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'white'};
  }
`;

const PageInfo = styled.div`
  font-size: 0.8rem;
  color: #718096;
  display: flex;
  align-items: center;
  gap: 4px;
`;

const AddWatermarkSection = styled.div`
  text-align: center;
  padding: ${props => props.isCompact ? '16px' : '30px'};
  border: 2px dashed #e2e8f0;
  border-radius: ${props => props.isCompact ? '8px' : '12px'};
  margin: ${props => props.isCompact ? '12px 24px 24px' : '20px 0 0'};
  transition: all 0.3s ease;

  &:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
  }
`;

function WatermarkConfig({ 
  watermarks, 
  onAddWatermark, 
  onRemoveWatermark, 
  onUpdateProperty, 
  onApplyWatermarks,
  onShowVisualPositioning,
  pdfInfo = null,
  isCompact = false
}) {
  const [colorPickers, setColorPickers] = React.useState({});

  const toggleColorPicker = (watermarkId) => {
    setColorPickers(prev => ({
      ...prev,
      [watermarkId]: !prev[watermarkId]
    }));
  };

  const handleColorChange = (watermarkId, color) => {
    onUpdateProperty(watermarkId, 'color', color.hex);
  };

  const handlePageSelection = (watermarkId, pageSelection) => {
    onUpdateProperty(watermarkId, 'target_pages', pageSelection);
  };

  const getSelectedPages = (watermark) => {
    return watermark.target_pages || [1]; // Default to first page
  };

  const isPageSelected = (watermark, pageNum) => {
    const selected = getSelectedPages(watermark);
    return selected === 'all' || (Array.isArray(selected) && selected.includes(pageNum));
  };

  const togglePageSelection = (watermark, pageNum) => {
    const selected = getSelectedPages(watermark);
    
    if (selected === 'all') {
      // If 'all' is selected, switch to just this page
      handlePageSelection(watermark.id, [pageNum]);
    } else if (Array.isArray(selected)) {
      if (selected.includes(pageNum)) {
        // Remove this page from selection
        const newSelection = selected.filter(p => p !== pageNum);
        handlePageSelection(watermark.id, newSelection.length > 0 ? newSelection : [1]);
      } else {
        // Add this page to selection
        handlePageSelection(watermark.id, [...selected, pageNum].sort());
      }
    }
  };

  const toggleAllPages = (watermark) => {
    const selected = getSelectedPages(watermark);
    if (selected === 'all') {
      handlePageSelection(watermark.id, [1]); // Switch to first page only
    } else {
      handlePageSelection(watermark.id, 'all'); // Select all pages
    }
  };

  return (
    <ConfigContainer>
      {!isCompact && (
        <Header>
          <Title>
            <Settings size={24} />
            Watermark Configuration
          </Title>
          <ActionButtons>
            <Button
              className="secondary"
              onClick={onShowVisualPositioning}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Eye size={16} />
              Visual Positioning
            </Button>
            <Button
              className="primary"
              onClick={onApplyWatermarks}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Settings size={16} />
              Apply Watermarks
            </Button>
          </ActionButtons>
        </Header>
      )}
      
      {isCompact && (
        <>
          <div style={{ padding: '16px 24px', borderBottom: '1px solid #e2e8f0', background: '#f8fafc' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', fontSize: '0.9rem', color: '#4a5568', marginBottom: '12px' }}>
              <Settings size={16} />
              Watermark Settings
            </div>
            <Button
              className="primary"
              onClick={onApplyWatermarks}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              style={{ width: '100%', fontSize: '0.9rem', padding: '10px 16px' }}
            >
              <Settings size={14} />
              Apply Watermarks
            </Button>
          </div>
        </>
      )}

      <WatermarksList isCompact={isCompact}>
        {watermarks.map((watermark, index) => (
          <WatermarkCard
            key={watermark.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            isCompact={isCompact}
          >
            <WatermarkHeader isCompact={isCompact}>
              <WatermarkTitle isCompact={isCompact}>
                <Type size={isCompact ? 16 : 20} />
                Watermark {index + 1}
              </WatermarkTitle>
              {watermarks.length > 1 && (
                <RemoveButton
                  onClick={() => onRemoveWatermark(watermark.id)}
                  title="Remove watermark"
                >
                  <X size={14} />
                </RemoveButton>
              )}
            </WatermarkHeader>

            <ConfigGrid isCompact={isCompact}>
              <ConfigGroup>
                <Label>Watermark Text</Label>
                <Input
                  type="text"
                  value={watermark.text}
                  onChange={(e) => onUpdateProperty(watermark.id, 'text', e.target.value)}
                  placeholder="Enter watermark text..."
                />
              </ConfigGroup>

              <ConfigGroup>
                <Label>Font Size</Label>
                <RangeContainer>
                  <RangeInput
                    type="range"
                    min="8"
                    max="72"
                    value={watermark.font_size}
                    onChange={(e) => onUpdateProperty(watermark.id, 'font_size', parseInt(e.target.value))}
                  />
                  <RangeValue>{watermark.font_size}</RangeValue>
                </RangeContainer>
              </ConfigGroup>

              <ConfigGroup>
                <Label>Color</Label>
                <ColorContainer>
                  <ColorPicker>
                    <ColorButton
                      color={watermark.color}
                      onClick={() => toggleColorPicker(watermark.id)}
                    />
                    {colorPickers[watermark.id] && (
                      <ColorPickerPopover>
                        <ColorPickerCover onClick={() => toggleColorPicker(watermark.id)} />
                        <ChromePicker
                          color={watermark.color}
                          onChange={(color) => handleColorChange(watermark.id, color)}
                        />
                      </ColorPickerPopover>
                    )}
                  </ColorPicker>
                  <Input
                    type="text"
                    value={watermark.color}
                    onChange={(e) => onUpdateProperty(watermark.id, 'color', e.target.value)}
                    placeholder="#000000"
                  />
                </ColorContainer>
              </ConfigGroup>

              <ConfigGroup>
                <Label>Opacity</Label>
                <RangeContainer>
                  <RangeInput
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={watermark.opacity}
                    onChange={(e) => onUpdateProperty(watermark.id, 'opacity', parseFloat(e.target.value))}
                  />
                  <RangeValue>{watermark.opacity}</RangeValue>
                </RangeContainer>
              </ConfigGroup>

              <ConfigGroup>
                <Label>Rotation (degrees)</Label>
                <RangeContainer>
                  <RangeInput
                    type="range"
                    min="0"
                    max="360"
                    value={watermark.rotation}
                    onChange={(e) => onUpdateProperty(watermark.id, 'rotation', parseInt(e.target.value))}
                  />
                  <RangeValue>{watermark.rotation}°</RangeValue>
                </RangeContainer>
              </ConfigGroup>

              {pdfInfo && pdfInfo.num_pages > 1 && (
                <ConfigGroup style={{ gridColumn: 'span 2' }}>
                  <Label>Target Pages</Label>
                  <PageSelector>
                    <PageInfo>
                      <FileText size={14} />
                      PDF has {pdfInfo.num_pages} pages
                    </PageInfo>
                    <PageOptions>
                      <PageOption
                        className="all-pages"
                        selected={getSelectedPages(watermark) === 'all'}
                        onClick={() => toggleAllPages(watermark)}
                      >
                        <CheckSquare size={14} />
                        All Pages
                      </PageOption>
                      {Array.from({ length: pdfInfo.num_pages }, (_, i) => i + 1).map(pageNum => (
                        <PageOption
                          key={pageNum}
                          selected={isPageSelected(watermark, pageNum)}
                          onClick={() => togglePageSelection(watermark, pageNum)}
                        >
                          Page {pageNum}
                        </PageOption>
                      ))}
                    </PageOptions>
                  </PageSelector>
                </ConfigGroup>
              )}
            </ConfigGrid>
          </WatermarkCard>
        ))}
      </WatermarksList>

      <AddWatermarkSection isCompact={isCompact}>
        <Button
          className="secondary"
          onClick={onAddWatermark}
          whileHover={{ scale: isCompact ? 1.02 : 1.05 }}
          whileTap={{ scale: 0.95 }}
          style={isCompact ? { width: '100%', fontSize: '0.9rem', padding: '10px 16px' } : {}}
        >
          <Plus size={isCompact ? 16 : 20} />
          Add Another Watermark
        </Button>
      </AddWatermarkSection>
    </ConfigContainer>
  );
}

export default WatermarkConfig;
