import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { Plus, X, Eye, Settings, Palette, Type, RotateCcw } from 'lucide-react';
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
  gap: 20px;
`;

const WatermarkCard = styled(motion.div)`
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 25px;
  position: relative;

  &:hover {
    border-color: #cbd5e0;
  }
`;

const WatermarkHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
`;

const WatermarkTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 600;
  color: #1a202c;
  display: flex;
  align-items: center;
  gap: 8px;
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
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
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

const AddWatermarkSection = styled.div`
  text-align: center;
  padding: 30px;
  border: 2px dashed #e2e8f0;
  border-radius: 12px;
  margin-top: 20px;
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
  onShowVisualPositioning 
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

  return (
    <ConfigContainer>
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

      <WatermarksList>
        {watermarks.map((watermark, index) => (
          <WatermarkCard
            key={watermark.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <WatermarkHeader>
              <WatermarkTitle>
                <Type size={20} />
                Watermark {index + 1}
              </WatermarkTitle>
              {watermarks.length > 1 && (
                <RemoveButton
                  onClick={() => onRemoveWatermark(watermark.id)}
                  title="Remove watermark"
                >
                  <X size={16} />
                </RemoveButton>
              )}
            </WatermarkHeader>

            <ConfigGrid>
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
            </ConfigGrid>
          </WatermarkCard>
        ))}
      </WatermarksList>

      <AddWatermarkSection>
        <Button
          className="secondary"
          onClick={onAddWatermark}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Plus size={20} />
          Add Another Watermark
        </Button>
      </AddWatermarkSection>
    </ConfigContainer>
  );
}

export default WatermarkConfig;
