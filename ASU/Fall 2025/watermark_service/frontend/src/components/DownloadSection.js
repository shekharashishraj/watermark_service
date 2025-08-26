import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { Download, CheckCircle, RotateCcw, FileText } from 'lucide-react';

const DownloadContainer = styled.div`
  text-align: center;
`;

const SuccessMessage = styled.div`
  background: #f0fff4;
  border: 2px solid #9ae6b4;
  border-radius: 12px;
  padding: 30px;
  margin-bottom: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
`;

const SuccessIcon = styled.div`
  color: #38a169;
  font-size: 2rem;
`;

const SuccessText = styled.div`
  text-align: left;
`;

const SuccessTitle = styled.h3`
  font-size: 1.3rem;
  font-weight: 600;
  color: #38a169;
  margin-bottom: 5px;
`;

const SuccessDescription = styled.p`
  color: #4a5568;
  margin: 0;
`;

const FileInfo = styled.div`
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
`;

const FileIcon = styled.div`
  color: #667eea;
  font-size: 2rem;
`;

const FileDetails = styled.div`
  text-align: left;
`;

const FileName = styled.h4`
  font-size: 1.1rem;
  font-weight: 600;
  color: #1a202c;
  margin-bottom: 5px;
`;

const FileStatus = styled.p`
  color: #718096;
  margin: 0;
  font-size: 0.9rem;
`;

const ActionButtons = styled.div`
  display: flex;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
`;

const Button = styled(motion.button)`
  padding: 15px 30px;
  border: none;
  border-radius: 10px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: all 0.3s ease;
  min-width: 160px;
  justify-content: center;

  &.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    
    &:hover {
      transform: translateY(-3px);
      box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
  }

  &.secondary {
    background: #f7fafc;
    color: #4a5568;
    border: 2px solid #e2e8f0;
    
    &:hover {
      background: #edf2f7;
      border-color: #cbd5e0;
      transform: translateY(-2px);
    }
  }

  &.danger {
    background: #fed7d7;
    color: #c53030;
    border: 2px solid #feb2b2;
    
    &:hover {
      background: #feb2b2;
      transform: translateY(-2px);
    }
  }
`;

const FeaturesList = styled.div`
  background: #ebf8ff;
  border: 1px solid #bee3f8;
  border-radius: 12px;
  padding: 25px;
  margin-top: 30px;
`;

const FeaturesTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 600;
  color: #2b6cb0;
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
`;

const FeatureItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: #4a5568;
  font-size: 0.9rem;

  &:before {
    content: "✓";
    color: #38a169;
    font-weight: bold;
    font-size: 1rem;
  }
`;

function DownloadSection({ outputFile, onDownload, onReset }) {
  return (
    <DownloadContainer>
      <SuccessMessage>
        <SuccessIcon>
          <CheckCircle size={40} />
        </SuccessIcon>
        <SuccessText>
          <SuccessTitle>Watermarks Applied Successfully!</SuccessTitle>
          <SuccessDescription>
            Your PDF has been processed and is ready for download. All watermarks have been applied with the specified settings.
          </SuccessDescription>
        </SuccessText>
      </SuccessMessage>

      <FileInfo>
        <FileIcon>
          <FileText size={40} />
        </FileIcon>
        <FileDetails>
          <FileName>{outputFile}</FileName>
          <FileStatus>Watermarked PDF ready for download</FileStatus>
        </FileDetails>
      </FileInfo>

      <ActionButtons>
        <Button
          className="primary"
          onClick={onDownload}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Download size={20} />
          Download PDF
        </Button>
        
        <Button
          className="secondary"
          onClick={onReset}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw size={20} />
          Process New File
        </Button>
      </ActionButtons>

      <FeaturesList>
        <FeaturesTitle>
          <FileText size={20} />
          What's Included in Your Watermarked PDF
        </FeaturesTitle>
        <FeaturesGrid>
          <FeatureItem>Professional watermark positioning</FeatureItem>
          <FeatureItem>Custom text and styling</FeatureItem>
          <FeatureItem>Multiple watermark support</FeatureItem>
          <FeatureItem>High-quality PDF output</FeatureItem>
          <FeatureItem>Preserved original content</FeatureItem>
          <FeatureItem>Ready for professional use</FeatureItem>
        </FeaturesGrid>
      </FeaturesList>
    </DownloadContainer>
  );
}

export default DownloadSection;
