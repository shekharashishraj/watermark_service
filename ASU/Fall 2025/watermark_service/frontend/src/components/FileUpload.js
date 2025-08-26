import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { Upload, FileText, AlertCircle } from 'lucide-react';

const UploadContainer = styled.div`
  text-align: center;
`;

const Dropzone = styled(motion.div)`
  border: 3px dashed ${props => props.isDragActive ? '#667eea' : '#e2e8f0'};
  border-radius: 16px;
  padding: 60px 40px;
  background: ${props => props.isDragActive ? 'rgba(102, 126, 234, 0.05)' : '#f8fafc'};
  cursor: pointer;
  transition: all 0.3s ease;
  margin-bottom: 30px;

  &:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
  }
`;

const UploadIcon = styled.div`
  font-size: 3rem;
  color: #667eea;
  margin-bottom: 20px;
`;

const UploadTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1a202c;
  margin-bottom: 10px;
`;

const UploadText = styled.p`
  font-size: 1rem;
  color: #718096;
  margin-bottom: 20px;
`;

const FileInfo = styled.div`
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  margin-top: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
`;

const FileIcon = styled.div`
  color: #667eea;
  font-size: 2rem;
`;

const FileDetails = styled.div`
  text-align: left;
`;

const FileName = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  color: #1a202c;
  margin-bottom: 5px;
`;

const FileSize = styled.p`
  font-size: 0.9rem;
  color: #718096;
`;

const ErrorMessage = styled.div`
  background: #fed7d7;
  border: 1px solid #feb2b2;
  border-radius: 8px;
  padding: 15px;
  margin-top: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  color: #c53030;
`;

const Instructions = styled.div`
  background: #ebf8ff;
  border: 1px solid #bee3f8;
  border-radius: 12px;
  padding: 20px;
  margin-top: 20px;
`;

const InstructionsTitle = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  color: #2b6cb0;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const InstructionsList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const InstructionItem = styled.li`
  color: #4a5568;
  margin-bottom: 8px;
  padding-left: 20px;
  position: relative;

  &:before {
    content: "•";
    color: #667eea;
    font-weight: bold;
    position: absolute;
    left: 0;
  }
`;

function FileUpload({ onFileUploaded, apiBaseUrl, showLoading, hideLoading }) {
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    
    if (!file) return;

    if (file.type !== 'application/pdf') {
      return;
    }

    showLoading('Uploading file...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiBaseUrl}/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        onFileUploaded(data.file_id, data.filename);
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      hideLoading();
    }
  }, [onFileUploaded, apiBaseUrl, showLoading, hideLoading]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: false
  });

  return (
    <UploadContainer>
      <Dropzone
        {...getRootProps()}
        isDragActive={isDragActive}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input {...getInputProps()} />
        <UploadIcon>
          <Upload size={60} />
        </UploadIcon>
        <UploadTitle>
          {isDragActive ? 'Drop your PDF here' : 'Upload PDF File'}
        </UploadTitle>
        <UploadText>
          {isDragActive 
            ? 'Release to upload your PDF'
            : 'Drag and drop a PDF file here, or click to select'
          }
        </UploadText>
      </Dropzone>

      {isDragReject && (
        <ErrorMessage>
          <AlertCircle size={20} />
          <span>Please upload a valid PDF file</span>
        </ErrorMessage>
      )}

      <Instructions>
        <InstructionsTitle>
          <FileText size={20} />
          How to use the PDF Watermark Service
        </InstructionsTitle>
        <InstructionsList>
          <InstructionItem>Upload a PDF file using drag-and-drop or click to browse</InstructionItem>
          <InstructionItem>Add watermarks with custom text, size, color, and opacity</InstructionItem>
          <InstructionItem>Use the visual positioning tool to drag watermarks to exact locations</InstructionItem>
          <InstructionItem>Apply watermarks and download your processed PDF</InstructionItem>
        </InstructionsList>
      </Instructions>
    </UploadContainer>
  );
}

export default FileUpload;
