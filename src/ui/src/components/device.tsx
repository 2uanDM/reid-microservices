import { DeviceStreamProps } from "@/types";
import { useEffect, useRef } from "react";
import { Camera, Users, Wifi, WifiOff } from "lucide-react";


export function DeviceStream({ deviceId, frameData, isConnected }: DeviceStreamProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
  
    useEffect(() => {
      if (!frameData || !canvasRef.current) return;
  
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
      img.src = `data:image/jpeg;base64,${frameData.image_base64}`;
    }, [frameData]);
  
  
    // const getLatencyMs = (frameData: FrameData) => {
    //   return Math.round((frameData.processed_at - frameData.created_at) / 1000000);
    // };
  
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
              {/*}
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-orange-500" />
                <span className="text-gray-600">Latency:</span>
                <span className="font-semibold text-gray-900">
                  {getLatencyMs(frameData)}ms
                </span>
              </div>
              */}
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