"use client";

import { useEffect, useState, useRef } from "react";
import { Camera, Users, Clock, Wifi, WifiOff } from "lucide-react";

interface TrackedPerson {
  person_id: number;
  bbox: number[];
  confidence: number;
  gender: string;
  gender_confidence: number;
}

interface FrameData {
  device_id: string;
  frame_number: number;
  tracked_persons: TrackedPerson[];
  created_at: number;
  processed_at: number;
  image_base64: string;
}

interface DeviceStreamProps {
  deviceId: string;
  frameData: FrameData | null;
  isConnected: boolean;
}

function DeviceStream({ deviceId, frameData, isConnected }: DeviceStreamProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!frameData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Draw the image
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frameData.image_base64}`;
  }, [frameData]);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp / 1000000).toLocaleTimeString();
  };

  const getLatencyMs = (frameData: FrameData) => {
    return Math.round((frameData.processed_at - frameData.created_at) / 1000000);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gray-50 px-4 py-3 border-b flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-5 h-5 text-gray-600" />
          <h3 className="font-semibold text-gray-900">{deviceId}</h3>
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-500" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-500" />
          )}
        </div>
        {frameData && (
          <div className="text-sm text-gray-600">
            Frame #{frameData.frame_number}
          </div>
        )}
      </div>

      {/* Video Stream */}
      <div className="relative bg-black">
        <canvas
          ref={canvasRef}
          className="w-full h-auto max-h-96 object-contain"
          style={{ display: frameData ? "block" : "none" }}
        />
        {!frameData && (
          <div className="aspect-video flex items-center justify-center text-gray-400">
            <div className="text-center">
              <Camera className="w-16 h-16 mx-auto mb-2 opacity-50" />
              <p>No video stream</p>
            </div>
          </div>
        )}
      </div>

      {/* Stats */}
      {frameData && (
        <div className="px-4 py-3 bg-gray-50 border-t">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <Users className="w-4 h-4 text-blue-500" />
              <span className="text-gray-600">Persons:</span>
              <span className="font-semibold text-gray-900">
                {frameData.tracked_persons.length}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-orange-500" />
              <span className="text-gray-600">Latency:</span>
              <span className="font-semibold text-gray-900">
                {getLatencyMs(frameData)}ms
              </span>
            </div>
          </div>
          
          {/* Person Details */}
          {frameData.tracked_persons.length > 0 && (
            <div className="mt-3 pt-3 border-t">
              <h4 className="text-xs font-semibold text-gray-700 mb-2">
                Tracked Persons:
              </h4>
              <div className="space-y-1">
                {frameData.tracked_persons.map((person) => (
                  <div
                    key={person.person_id}
                    className="text-xs bg-white rounded px-2 py-1 flex justify-between items-center"
                  >
                    <span className="font-medium">ID: {person.person_id}</span>
                    <span className="text-gray-600">
                      {person.gender} ({Math.round(person.gender_confidence * 100)}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [devices, setDevices] = useState<string[]>([]);
  const [frameData, setFrameData] = useState<Map<string, FrameData>>(new Map());
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket streaming service
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket("ws://localhost:8765");
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("Connected to streaming service");
          setIsConnected(true);
        };

        ws.onmessage = (event) => {
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
        };

        ws.onclose = () => {
          console.log("Disconnected from streaming service");
          setIsConnected(false);
          
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error("WebSocket error:", error);
          setIsConnected(false);
        };
        
      } catch (error) {
        console.error("Failed to connect to WebSocket:", error);
        setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                ReID Video Dashboard
              </h1>
              <p className="text-sm text-gray-600">
                Real-time person re-identification monitoring
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {isConnected ? (
                  <Wifi className="w-5 h-5 text-green-500" />
                ) : (
                  <WifiOff className="w-5 h-5 text-red-500" />
                )}
                <span className="text-sm text-gray-600">
                  {isConnected ? "Connected" : "Disconnected"}
                </span>
              </div>
              
              <div className="text-sm text-gray-600">
                {devices.length} device{devices.length !== 1 ? "s" : ""} active
              </div>
            </div>
          </div>
        </div>
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
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {devices.map((deviceId) => (
              <DeviceStream
                key={deviceId}
                deviceId={deviceId}
                frameData={frameData.get(deviceId) || null}
                isConnected={isConnected}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
