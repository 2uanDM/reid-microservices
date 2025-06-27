import { DeviceStreamProps, FrameData } from "@/types";
import { useEffect, useRef, useState, useCallback } from "react";
import { Camera, Users, Wifi, WifiOff, RotateCcw, Settings } from "lucide-react";


export function DeviceStream({ 
  deviceId, 
  frameData, 
  isConnected, 
  onReconnect, 
  fps, 
  onFpsChange 
}: DeviceStreamProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const frameBufferRef = useRef<FrameData[]>([]);
    const animationFrameRef = useRef<number>(0);
    const lastFrameTimeRef = useRef<number>(0);
    const [currentFrame, setCurrentFrame] = useState<FrameData | null>(null);
    const [bufferedFrames, setBufferedFrames] = useState<number>(0);
    const [showSettings, setShowSettings] = useState(false);
    
    // Calculate frame interval based on FPS
    const frameInterval = 1000 / fps; // milliseconds per frame
    
    // Buffer management
    const bufferSize = Math.max(fps * 2, 30); // Keep at least 2 seconds worth of frames
    
    // Add new frame to buffer
    useEffect(() => {
      if (frameData) {
        frameBufferRef.current.push(frameData);
        
        // Keep buffer size manageable
        if (frameBufferRef.current.length > bufferSize) {
          frameBufferRef.current.shift();
        }
        
        setBufferedFrames(frameBufferRef.current.length);
      }
    }, [frameData, bufferSize]);
    
    // Frame playback logic
    const playNextFrame = useCallback(() => {
      const now = performance.now();
      
      if (now - lastFrameTimeRef.current >= frameInterval && frameBufferRef.current.length > 0) {
        const nextFrame = frameBufferRef.current.shift();
        if (nextFrame) {
          setCurrentFrame(nextFrame);
          setBufferedFrames(frameBufferRef.current.length);
          lastFrameTimeRef.current = now;
        }
      }
      
      if (isConnected) {
        animationFrameRef.current = requestAnimationFrame(playNextFrame);
      }
    }, [frameInterval, isConnected]);
    
    // Start/stop playback
    useEffect(() => {
      if (isConnected) {
        animationFrameRef.current = requestAnimationFrame(playNextFrame);
      } else {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      }
      
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }, [isConnected, playNextFrame]);
    
    // Render current frame to canvas
    useEffect(() => {
      if (!currentFrame || !canvasRef.current) return;
  
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
  
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw the image
        ctx.drawImage(img, 0, 0);
      };
      img.src = `data:image/jpeg;base64,${currentFrame.image_base64}`;
    }, [currentFrame]);

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
          <div className="flex items-center space-x-2">
            {currentFrame && (
              <div className="text-sm text-gray-600">
                Frame #{currentFrame.frame_number}
              </div>
            )}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-1 rounded-md hover:bg-gray-200 transition-colors"
              title="Settings"
            >
              <Settings className="w-4 h-4 text-gray-600" />
            </button>
            <button
              onClick={onReconnect}
              className="p-1 rounded-md hover:bg-gray-200 transition-colors"
              title="Force Reconnect"
            >
              <RotateCcw className="w-4 h-4 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="bg-blue-50 px-4 py-3 border-b">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Playback Settings</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-600">FPS:</label>
                <input
                  type="range"
                  min="1"
                  max="60"
                  value={fps}
                  onChange={(e) => onFpsChange(parseInt(e.target.value))}
                  className="w-20"
                />
                <span className="text-sm font-medium text-gray-900 w-8">{fps}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Buffer:</span>
                <span className="text-sm font-medium text-gray-900">{bufferedFrames} frames</span>
              </div>
            </div>
          </div>
        )}

        {/* Video Stream */}
        <div className="relative bg-black">
          <canvas
            ref={canvasRef}
            className="w-full h-auto max-h-96 object-contain"
            style={{ display: currentFrame ? "block" : "none" }}
          />
          {!currentFrame && (
            <div className="aspect-video flex items-center justify-center text-gray-400">
              <div className="text-center">
                <Camera className="w-16 h-16 mx-auto mb-2 opacity-50" />
                <p>No video stream</p>
                {bufferedFrames > 0 && (
                  <p className="text-sm">Buffering... ({bufferedFrames} frames)</p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Stats */}
        {currentFrame && (
          <div className="px-4 py-3 bg-gray-50 border-t">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-blue-500" />
                <span className="text-gray-600">Persons:</span>
                <span className="font-semibold text-gray-900">
                  {currentFrame.tracked_persons.length}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-gray-600">FPS:</span>
                <span className="font-semibold text-gray-900">{fps}</span>
              </div>
            </div>
            
            {/* Person Details */}
            {currentFrame.tracked_persons.length > 0 && (
              <div className="mt-3 pt-3 border-t">
                <h4 className="text-xs font-semibold text-gray-700 mb-2">
                  Tracked Persons:
                </h4>
                <div className="space-y-1">
                  {currentFrame.tracked_persons.map((person) => (
                    <div
                      key={person.person_id}
                      className="text-xs bg-white rounded px-2 py-1 flex justify-between items-center"
                    >
                      <span className="font-medium text-gray-900">ID: {person.person_id}</span>
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