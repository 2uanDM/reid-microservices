"use client";

import { useEffect, useState, useRef } from "react";
import { Camera, Wifi, WifiOff } from "lucide-react";
import { FrameData } from "@/types";
import { DeviceStream } from "@/components/device";

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [devices, setDevices] = useState<string[]>([]);
  const [frameData, setFrameData] = useState<Map<string, FrameData>>(new Map());
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket streaming service
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket("ws://quandm.myvnc.com:8765");
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
          
          // Attempt to reconnect after 2 seconds
          setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = (error) => {
          console.error("WebSocket error:", error);
          setIsConnected(false);
        };
        
      } catch (error) {
        console.error("Failed to connect to WebSocket", error);
        setTimeout(connectWebSocket, 2000);
      }
    };

    connectWebSocket();

    // NOTE: For multi-instance streaming services, you would need:
    // const streamingUrls = [
    //   "ws://localhost:8765",
    //   "ws://localhost:8766", 
    //   "ws://localhost:8767"
    // ];
    // Connect to all URLs and merge device data from all connections

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
                Re-ID Video Dashboard
              </h1>
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
