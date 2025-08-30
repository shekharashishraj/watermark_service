/**
 * Frontend Logging Utility
 * Provides centralized logging for error tracking, performance monitoring, and debugging
 */

class Logger {
  constructor() {
    this.logs = [];
    this.maxLogs = 1000; // Keep last 1000 logs in memory
    this.isDebugMode = process.env.NODE_ENV !== 'production';
    this.sessionId = this.generateSessionId();
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  createLogEntry(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      data,
      sessionId: this.sessionId,
      url: window.location.href,
      userAgent: navigator.userAgent
    };

    // Add to in-memory logs
    this.logs.push(logEntry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift(); // Remove oldest log
    }

    return logEntry;
  }

  // Standard logging methods
  info(message, data = null) {
    const logEntry = this.createLogEntry('INFO', message, data);
    if (this.isDebugMode) {
      console.log(`[INFO] ${message}`, data || '');
    }
    return logEntry;
  }

  warn(message, data = null) {
    const logEntry = this.createLogEntry('WARN', message, data);
    console.warn(`[WARN] ${message}`, data || '');
    return logEntry;
  }

  error(message, error = null, data = null) {
    const errorData = {
      ...data,
      error: error ? {
        message: error.message,
        stack: error.stack,
        name: error.name
      } : null
    };
    
    const logEntry = this.createLogEntry('ERROR', message, errorData);
    console.error(`[ERROR] ${message}`, errorData);
    return logEntry;
  }

  debug(message, data = null) {
    if (!this.isDebugMode) return null;
    
    const logEntry = this.createLogEntry('DEBUG', message, data);
    console.debug(`[DEBUG] ${message}`, data || '');
    return logEntry;
  }

  // Performance monitoring
  startPerformanceTimer(name) {
    const startTime = performance.now();
    this.info(`Performance timer started: ${name}`);
    
    return {
      end: () => {
        const endTime = performance.now();
        const duration = endTime - startTime;
        this.info(`Performance timer ended: ${name}`, { duration: `${duration.toFixed(2)}ms` });
        return duration;
      }
    };
  }

  // WebSocket connection logging
  logWebSocketEvent(eventType, data = null) {
    this.info(`WebSocket ${eventType}`, data);
  }

  // Component lifecycle logging
  logComponentMount(componentName, props = null) {
    this.debug(`Component mounted: ${componentName}`, { props });
  }

  logComponentUnmount(componentName) {
    this.debug(`Component unmounted: ${componentName}`);
  }

  // User interaction logging
  logUserInteraction(action, element = null, data = null) {
    this.info(`User interaction: ${action}`, { element, ...data });
  }

  // File operation logging
  logFileOperation(operation, filename, fileSize = null, duration = null) {
    this.info(`File operation: ${operation}`, {
      filename,
      fileSize: fileSize ? `${(fileSize / 1024).toFixed(2)} KB` : null,
      duration: duration ? `${duration.toFixed(2)}ms` : null
    });
  }

  // Watermark operation logging
  logWatermarkOperation(operation, watermarkId, properties = null) {
    this.info(`Watermark operation: ${operation}`, {
      watermarkId,
      properties
    });
  }

  // Export logs for debugging or support
  exportLogs(filterLevel = null) {
    const filteredLogs = filterLevel 
      ? this.logs.filter(log => log.level === filterLevel)
      : this.logs;

    return {
      sessionId: this.sessionId,
      exportedAt: new Date().toISOString(),
      totalLogs: filteredLogs.length,
      logs: filteredLogs
    };
  }

  // Clear logs
  clearLogs() {
    const previousCount = this.logs.length;
    this.logs = [];
    this.info(`Logs cleared`, { previousCount });
  }

  // Get summary statistics
  getLogsSummary() {
    const summary = {
      total: this.logs.length,
      errors: this.logs.filter(log => log.level === 'ERROR').length,
      warnings: this.logs.filter(log => log.level === 'WARN').length,
      info: this.logs.filter(log => log.level === 'INFO').length,
      debug: this.logs.filter(log => log.level === 'DEBUG').length,
      sessionId: this.sessionId,
      oldestLog: this.logs.length > 0 ? this.logs[0].timestamp : null,
      newestLog: this.logs.length > 0 ? this.logs[this.logs.length - 1].timestamp : null
    };

    return summary;
  }

  // Send logs to backend for persistent storage (optional)
  async sendLogsToBackend(logsToSend = null) {
    if (!logsToSend) {
      logsToSend = this.logs.slice(-50); // Send last 50 logs
    }

    try {
      const response = await fetch('/api/frontend-logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          logs: logsToSend
        })
      });

      if (response.ok) {
        this.info(`Sent ${logsToSend.length} logs to backend`);
      } else {
        this.warn('Failed to send logs to backend', { status: response.status });
      }
    } catch (error) {
      this.warn('Error sending logs to backend', { error: error.message });
    }
  }
}

// Create singleton instance
const logger = new Logger();

// Log initial page load
logger.info('Frontend logger initialized', {
  url: window.location.href,
  userAgent: navigator.userAgent,
  screenResolution: window.screen ? `${window.screen.width}x${window.screen.height}` : 'unknown',
  timestamp: new Date().toISOString()
});

// Automatically log unhandled errors
window.addEventListener('error', (event) => {
  logger.error('Unhandled JavaScript error', event.error, {
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno
  });
});

// Log unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  logger.error('Unhandled promise rejection', null, {
    reason: event.reason
  });
});

// Expose logger globally for debugging
if (typeof window !== 'undefined') {
  window._logger = logger;
}

export default logger;