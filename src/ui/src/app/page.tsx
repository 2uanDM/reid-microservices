"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Camera, Wifi, WifiOff, RotateCcw, Settings } from "lucide-react";
import { FrameData } from "@/types";
import { DeviceStream } from "@/components/device";

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [devices, setDevices] = useState<string[]>([]);
  const [frameData, setFrameData] = useState<Map<string, FrameData>>(new Map());
  const [fps, setFps] = useState(12); // Default FPS
  const [showGlobalSettings, setShowGlobalSettings] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000; // Start with 1 second

  // Calculate exponential backoff delay
  const getReconnectDelay = () => {
    const delay = Math.min(baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current), 30000); // Max 30 seconds
    return delay;
  };

  const cleanupConnection = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      
      if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
        wsRef.current.close(1000, "Client disconnecting");
      }
      wsRef.current = null;
    }
  }, []);

  const connectWebSocket = useCallback((isManualReconnect = false) => {
    // Prevent multiple concurrent connection attempts
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      console.log("Connection already exists or in progress");
      return;
    }

    // Clean up any existing connection
    cleanupConnection();

    if (isManualReconnect) {
      reconnectAttemptsRef.current = 0;
    }

    // Check if we've exceeded max reconnect attempts
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.log("Max reconnect attempts reached. Stopping automatic reconnection.");
      setConnectionStatus('disconnected');
      return;
    }

    try {
      setConnectionStatus(reconnectAttemptsRef.current === 0 ? 'connecting' : 'reconnecting');
      console.log(`Connecting to WebSocket (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})...`);
      
      const ws = new WebSocket("ws://localhost:8765");
      wsRef.current = ws;

      // Set connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          console.log("Connection timeout, closing...");
          ws.close();
        }
      }, 10000); // 10 second timeout

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log("Connected to streaming service");
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0; // Reset attempts on successful connection
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case "device_list":
              setDevices(data.devices);
              break;
              
            case "frame_update":
              setFrameData(prev => {
                const newMap = new Map(prev);
                newMap.set(data.device_id, data);
                return newMap;
              });
              break;
              
            case "error":
              console.error("WebSocket error:", data.message);
              break;
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}`);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        
        // Only auto-reconnect if it wasn't a manual close and we haven't exceeded max attempts
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          const delay = getReconnectDelay();
          console.log(`Scheduling reconnection in ${delay}ms...`);
          
          setConnectionStatus('reconnecting');
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, delay);
        }
      };

      ws.onerror = (error) => {
        clearTimeout(connectionTimeout);
        console.error("WebSocket error:", error);
        setIsConnected(false);
        setConnectionStatus('disconnected');
      };
      
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
      setConnectionStatus('disconnected');
      
      // Schedule retry if we haven't exceeded max attempts
      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++;
        const delay = getReconnectDelay();
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      }
    }
  }, [cleanupConnection, maxReconnectAttempts, baseReconnectDelay]);

  const handleReconnect = useCallback(() => {
    console.log("Manual reconnection requested...");
    reconnectAttemptsRef.current = 0; // Reset attempts for manual reconnection
    connectWebSocket(true);
  }, [connectWebSocket]);

  const handleFpsChange = useCallback((newFps: number) => {
    setFps(newFps);
  }, []);

  useEffect(() => {
    connectWebSocket(true);

    return () => {
      cleanupConnection();
    };
  }, [connectWebSocket, cleanupConnection]);

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connecting':
        return "Connecting...";
      case 'connected':
        return "Connected";
      case 'reconnecting':
        return `Reconnecting... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`;
      case 'disconnected':
        return reconnectAttemptsRef.current >= maxReconnectAttempts ? "Connection failed" : "Disconnected";
      default:
        return "Unknown";
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connecting':
      case 'reconnecting':
        return "text-yellow-500";
      case 'connected':
        return "text-green-500";
      case 'disconnected':
        return "text-red-500";
      default:
        return "text-gray-500";
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Re-ID Video Dashboard
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {isConnected ? (
                  <Wifi className="w-5 h-5 text-green-500" />
                ) : connectionStatus === 'connecting' || connectionStatus === 'reconnecting' ? (
                  <Wifi className="w-5 h-5 text-yellow-500 animate-pulse" />
                ) : (
                  <WifiOff className="w-5 h-5 text-red-500" />
                )}
                <span className={`text-sm ${getConnectionStatusColor()}`}>
                  {getConnectionStatusText()}
                </span>
              </div>
              
              <div className="text-sm text-gray-600">
                {devices.length} device{devices.length !== 1 ? "s" : ""} active
              </div>

              <button
                onClick={() => setShowGlobalSettings(!showGlobalSettings)}
                className="p-2 rounded-md hover:bg-gray-100 transition-colors"
                title="Global Settings"
              >
                <Settings className="w-5 h-5 text-gray-600" />
              </button>

              <button
                onClick={handleReconnect}
                className="p-2 rounded-md hover:bg-gray-100 transition-colors"
                title="Force Reconnect"
                disabled={connectionStatus === 'connecting' || connectionStatus === 'reconnecting'}
              >
                <RotateCcw className={`w-5 h-5 ${
                  connectionStatus === 'connecting' || connectionStatus === 'reconnecting' 
                    ? 'text-gray-400 animate-spin' 
                    : 'text-gray-600'
                }`} />
              </button>
            </div>
          </div>
        </div>

        {/* Global Settings Panel */}
        {showGlobalSettings && (
          <div className="bg-blue-50 border-t">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-gray-900">Global Settings</h3>
              </div>
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-3">
                  <label className="text-sm font-medium text-gray-700">Default FPS:</label>
                  <input
                    type="range"
                    min="1"
                    max="60"
                    value={fps}
                    onChange={(e) => handleFpsChange(parseInt(e.target.value))}
                    className="w-32"
                  />
                  <span className="text-sm font-bold text-gray-900 w-8">{fps}</span>
                </div>
                <div className="text-sm text-gray-600">
                  This FPS setting applies to all device streams
                </div>
                <div className="text-xs text-gray-500">
                  Connection attempts: {reconnectAttemptsRef.current}/{maxReconnectAttempts}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {devices.length === 0 ? (
          <div className="text-center py-12">
            <Camera className="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No devices connected
            </h3>
            <p className="text-gray-600">
              Waiting for edge devices to start streaming...
            </p>
            {connectionStatus === 'reconnecting' && (
              <p className="text-sm text-yellow-600 mt-2">
                Attempting to reconnect to server...
              </p>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {devices.map((deviceId) => (
              <DeviceStream
                key={deviceId}
                deviceId={deviceId}
                frameData={frameData.get(deviceId) || null}
                isConnected={isConnected}
                onReconnect={handleReconnect}
                fps={fps}
                onFpsChange={handleFpsChange}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
