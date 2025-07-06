import { DeviceStreamProps, FrameData } from "@/types";
import { useEffect, useRef, useState, useCallback } from "react";
import { Camera, Users, Wifi, WifiOff, RotateCcw, Settings, Expand, X } from "lucide-react";


export function DeviceStream({ 
  deviceId, 
  frameData, 
  isConnected, 
  onReconnect, 
  fps, 
  onFpsChange 
}: DeviceStreamProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const modalCanvasRef = useRef<HTMLCanvasElement>(null);
    const frameBufferRef = useRef<FrameData[]>([]);
    const animationFrameRef = useRef<number>(0);
    const lastFrameTimeRef = useRef<number>(0);
    const [currentFrame, setCurrentFrame] = useState<FrameData | null>(null);
    const [bufferedFrames, setBufferedFrames] = useState<number>(0);
    const [showSettings, setShowSettings] = useState(true);
    const [isModalOpen, setIsModalOpen] = useState(false);
    
    // Calculate frame interval based on FPS
    const frameInterval = 1000 / fps; // milliseconds per frame
    
    // Buffer management
    const bufferSize = Math.max(fps * 2, 500); // Keep at least 2 seconds worth of frames
    
    // Modal controls
    const openModal = useCallback(() => {
      setIsModalOpen(true);
    }, []);
    
    const closeModal = useCallback(() => {
      setIsModalOpen(false);
    }, []);
    
    // Handle escape key to close modal
    useEffect(() => {
      const handleEscape = (e: KeyboardEvent) => {
        if (e.key === 'Escape' && isModalOpen) {
          closeModal();
        }
      };
      
      if (isModalOpen) {
        document.addEventListener('keydown', handleEscape);
        document.body.style.overflow = 'hidden'; // Prevent background scroll
      }
      
      return () => {
        document.removeEventListener('keydown', handleEscape);
        document.body.style.overflow = 'unset';
      };
    }, [isModalOpen, closeModal]);
    
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
    const renderFrame = useCallback((canvas: HTMLCanvasElement, frame: FrameData) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
  
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw the image
        ctx.drawImage(img, 0, 0);
      };
      img.src = `data:image/webp;base64,${frame.image_base64}`;
    }, []);
    
    useEffect(() => {
      if (!currentFrame || !canvasRef.current) return;
      renderFrame(canvasRef.current, currentFrame);
    }, [currentFrame, renderFrame]);
    
    // Render frame to modal canvas when modal is open
    useEffect(() => {
      if (!currentFrame || !modalCanvasRef.current || !isModalOpen) return;
      renderFrame(modalCanvasRef.current, currentFrame);
    }, [currentFrame, renderFrame, isModalOpen]);

    return (
      <>
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
            {/* Expand Button */}
            <button
              onClick={openModal}
              className="absolute top-2 right-2 z-10 p-2 bg-black bg-opacity-50 text-white rounded-md hover:bg-opacity-70 transition-all"
              title="Expand View"
            >
              <Expand className="w-4 h-4" />
            </button>
            
            <canvas
              ref={canvasRef}
              className="w-full h-auto max-h-96 object-contain cursor-pointer"
              style={{ display: currentFrame ? "block" : "none" }}
              onClick={openModal}
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

        {/* Modal */}
        {isModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75">
            <div className="relative max-w-7xl max-h-screen p-4">
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2 text-white">
                  <Camera className="w-6 h-6" />
                  <h2 className="text-xl font-semibold">{deviceId}</h2>
                  {isConnected ? (
                    <Wifi className="w-5 h-5 text-green-400" />
                  ) : (
                    <WifiOff className="w-5 h-5 text-red-400" />
                  )}
                  {currentFrame && (
                    <span className="text-sm text-gray-300">
                      Frame #{currentFrame.frame_number}
                    </span>
                  )}
                </div>
                <button
                  onClick={closeModal}
                  className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-md transition-colors"
                  title="Close (Esc)"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Modal Content */}
              <div className="bg-black rounded-lg overflow-hidden">
                <canvas
                  ref={modalCanvasRef}
                  className="max-w-full max-h-[80vh] object-contain"
                  style={{ display: currentFrame ? "block" : "none" }}
                />
                {!currentFrame && (
                  <div className="aspect-video flex items-center justify-center text-gray-400 min-h-[400px]">
                    <div className="text-center">
                      <Camera className="w-24 h-24 mx-auto mb-4 opacity-50" />
                      <p className="text-lg">No video stream</p>
                      {bufferedFrames > 0 && (
                        <p className="text-sm mt-2">Buffering... ({bufferedFrames} frames)</p>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Modal Stats */}
              {currentFrame && (
                <div className="mt-4 bg-gray-900 text-white p-4 rounded-lg">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <Users className="w-4 h-4 text-blue-400" />
                      <span className="text-gray-300">Persons:</span>
                      <span className="font-semibold">{currentFrame.tracked_persons.length}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-300">FPS:</span>
                      <span className="font-semibold">{fps}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-300">Buffer:</span>
                      <span className="font-semibold">{bufferedFrames} frames</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-300">Device:</span>
                      <span className="font-semibold">{deviceId}</span>
                    </div>
                  </div>
                  
                  {/* Person Details in Modal */}
                  {currentFrame.tracked_persons.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <h4 className="text-sm font-semibold text-gray-300 mb-2">
                        Tracked Persons:
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                        {currentFrame.tracked_persons.map((person) => (
                          <div
                            key={person.person_id}
                            className="text-xs bg-gray-800 rounded px-3 py-2 flex justify-between items-center"
                          >
                            <span className="font-medium">ID: {person.person_id}</span>
                            <span className="text-gray-400">
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
          </div>
        )}
      </>
    );
  }