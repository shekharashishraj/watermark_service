# API Documentation

## Overview
This document provides comprehensive API documentation for the Watermark Service backend endpoints and WebSocket events.

**Base URL**: `http://localhost:5001/api`  
**WebSocket URL**: `http://localhost:5001`  
**Protocol**: HTTP/1.1, WebSocket  
**Authentication**: None (Development)

---

## REST API Endpoints

### Health Check
**GET** `/api/health`

Check if the backend service is running and healthy.

**Response**:
```json
{
  "status": "healthy",
  "message": "PDF Watermark Service is running"
}
```

**Status Codes**:
- `200 OK`: Service is healthy

---

### File Upload
**POST** `/api/upload`

Upload a PDF file for watermarking.

**Content-Type**: `multipart/form-data`

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | PDF file to upload (max 16MB) |

**Response** (Success):
```json
{
  "success": true,
  "file_id": "uuid-string",
  "filename": "original-filename.pdf",
  "message": "File uploaded successfully"
}
```

**Response** (Error):
```json
{
  "error": "Error message description"
}
```

**Status Codes**:
- `200 OK`: File uploaded successfully
- `400 Bad Request`: No file provided or invalid file type
- `413 Payload Too Large`: File exceeds 16MB limit

**Supported File Types**: PDF (`.pdf`)

---

### Apply Watermark
**POST** `/api/watermark`

Apply watermarks to an uploaded PDF file.

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "file_id": "uuid-string",
  "watermarks": [
    {
      "id": "watermark-unique-id",
      "text": "CONFIDENTIAL",
      "position": "center", // or "custom"
      "font_size": 24,
      "color": "#FF0000",
      "opacity": 0.5,
      "rotation": 45,
      "custom_x": 100,  // if position is "custom"
      "custom_y": 200   // if position is "custom"
    }
  ]
}
```

**Watermark Object Properties**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | String | Yes | Unique watermark identifier |
| text | String | Yes | Text to display |
| position | String | Yes | "center" or "custom" |
| font_size | Number | No | Font size (default: 12) |
| color | String | No | Hex color (default: "#000000") |
| opacity | Number | No | Opacity 0-1 (default: 0.5) |
| rotation | Number | No | Rotation in degrees (default: 0) |
| custom_x | Number | Conditional | X position if position="custom" |
| custom_y | Number | Conditional | Y position if position="custom" |

**Response** (Success):
```json
{
  "success": true,
  "output_file": "watermarked_uuid.pdf",
  "message": "2 watermark(s) applied successfully"
}
```

**Response** (Error):
```json
{
  "error": "Error message description"
}
```

**Status Codes**:
- `200 OK`: Watermarks applied successfully
- `400 Bad Request`: No watermarks specified
- `404 Not Found`: File ID not found
- `500 Internal Server Error`: Processing error

---

### Download Watermarked File
**GET** `/api/download/{filename}`

Download a watermarked PDF file.

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| filename | String | Yes | Output filename from watermark response |

**Response**: Binary PDF file data

**Status Codes**:
- `200 OK`: File downloaded successfully
- `404 Not Found`: File not found

**Headers**:
- `Content-Type: application/pdf`
- `Content-Disposition: attachment; filename="filename.pdf"`

---

### Get Original File
**GET** `/api/original/{file_id}`

Get the original uploaded PDF for preview.

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file_id | String | Yes | UUID of uploaded file |

**Response**: Binary PDF file data

**Status Codes**:
- `200 OK`: File retrieved successfully
- `404 Not Found`: File ID not found

**Headers**:
- `Content-Type: application/pdf`

---

### Get PDF Preview
**GET** `/api/preview/{file_id}`

Get PDF preview for visual positioning interface.

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file_id | String | Yes | UUID of uploaded file |

**Response**: Binary PDF file data

**Status Codes**:
- `200 OK`: Preview retrieved successfully
- `404 Not Found`: File ID not found

**Headers**:
- `Content-Type: application/pdf`

---

### Cleanup Files
**POST** `/api/cleanup`

Clean up temporary files for a session.

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "file_id": "uuid-string"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Files cleaned up successfully"
}
```

**Status Codes**:
- `200 OK`: Cleanup completed
- `500 Internal Server Error`: Cleanup failed

---

## WebSocket Events

### Connection Events

#### `connect`
Triggered when client connects to WebSocket server.

**Server Response**:
```json
{
  "message": "Connected to watermark service"
}
```

#### `disconnect`
Triggered when client disconnects from WebSocket server.

**No response data**

---

### Session Management

#### `join_session`
Join a watermarking session for real-time collaboration.

**Client Request**:
```json
{
  "file_id": "uuid-string"
}
```

**Server Response**:
```json
{
  "file_id": "uuid-string",
  "watermarks": [
    // Array of current watermarks in session
  ]
}
```

#### `leave_session`
Leave a watermarking session.

**Client Request**:
```json
{
  "file_id": "uuid-string"
}
```

**Server Response**:
```json
{
  "file_id": "uuid-string"
}
```

---

### Watermark Operations

#### `add_watermark`
Add a new watermark to the session.

**Client Request**:
```json
{
  "file_id": "uuid-string",
  "watermark": {
    "id": "watermark-unique-id",
    "text": "Sample Text",
    "position": "center",
    // ... other watermark properties
  }
}
```

**Server Broadcast**:
```json
{
  "watermark": {
    // Complete watermark object
  }
}
```

#### `remove_watermark`
Remove a watermark from the session.

**Client Request**:
```json
{
  "file_id": "uuid-string",
  "watermark_id": "watermark-unique-id"
}
```

**Server Broadcast**:
```json
{
  "watermark_id": "watermark-unique-id"
}
```

#### `update_watermark_position`
Update watermark position in real-time (optimized for drag operations).

**Client Request**:
```json
{
  "file_id": "uuid-string",
  "watermark_id": "watermark-unique-id",
  "position": {
    "x": 150,
    "y": 200
  }
}
```

**Server Broadcast**:
```json
{
  "watermark_id": "watermark-unique-id",
  "position": {
    "x": 150,
    "y": 200
  }
}
```

#### `update_watermark_properties`
Update watermark properties (text, color, size, rotation, etc.).

**Client Request**:
```json
{
  "file_id": "uuid-string",
  "watermark_id": "watermark-unique-id",
  "properties": {
    "text": "Updated Text",
    "color": "#FF0000",
    "font_size": 28,
    "opacity": 0.7,
    "rotation": 45
  }
}
```

**Server Broadcast**:
```json
{
  "watermark_id": "watermark-unique-id",
  "properties": {
    // Updated properties object
  }
}
```

---

## Performance Optimizations

### Throttling and Debouncing
- **Position Updates**: Limited to 60fps (16.67ms intervals) for smooth UI
- **Server Synchronization**: Debounced 300ms after drag completion
- **Message Batching**: Maximum 10 server updates per second during active dragging

### Session Management
- **Room-based Updates**: Only clients in the same session receive updates
- **Memory Management**: Automatic session cleanup on disconnect
- **File Cleanup**: Temporary files removed when session ends

---

## Error Handling

### Common Error Responses

**File Not Found**:
```json
{
  "error": "File not found"
}
```

**Invalid File Type**:
```json
{
  "error": "Invalid file type"
}
```

**File Too Large**:
```json
{
  "error": "File size exceeds maximum limit (16MB)"
}
```

**Processing Error**:
```json
{
  "error": "Failed to process watermarks: [detailed error message]"
}
```

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Resource Not Found
- `413`: Payload Too Large
- `500`: Internal Server Error

---

## Client Libraries and Examples

### JavaScript/React WebSocket Client
```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:5001');

// Join session
socket.emit('join_session', { file_id: 'your-file-id' });

// Listen for watermark updates
socket.on('watermark_position_updated', (data) => {
  console.log('Watermark moved:', data);
});

// Update watermark position
socket.emit('update_watermark_position', {
  file_id: 'your-file-id',
  watermark_id: 'watermark-id',
  position: { x: 100, y: 200 }
});
```

### Fetch API Examples
```javascript
// Upload file
const formData = new FormData();
formData.append('file', pdfFile);

const uploadResponse = await fetch('/api/upload', {
  method: 'POST',
  body: formData
});

// Apply watermarks
const watermarkResponse = await fetch('/api/watermark', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    file_id: 'uuid-string',
    watermarks: [...]
  })
});
```

---

## Rate Limits and Quotas

**Development Environment**:
- No rate limits currently implemented
- File size limit: 16MB
- Session timeout: No automatic timeout
- Concurrent sessions: Unlimited

**Production Recommendations**:
- Implement rate limiting for upload endpoints
- Add authentication and user quotas  
- Set session timeouts and cleanup policies
- Monitor WebSocket connection limits

---

## Security Considerations

### Current Security Measures
- File type validation (PDF only)
- File size limits (16MB)
- Secure filename generation
- CORS enabled for localhost:3000

### Recommended Enhancements
- Input sanitization for watermark text
- File content validation beyond extension
- Authentication and authorization
- Request rate limiting
- Secure file storage with encryption