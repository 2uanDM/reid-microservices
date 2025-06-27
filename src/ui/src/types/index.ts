export interface TrackedPerson {
    person_id: number;
    bbox: number[];
    confidence: number;
    gender: string;
    gender_confidence: number;
  }
  
export interface FrameData {
  device_id: string;
  frame_number: number;
  tracked_persons: TrackedPerson[];
  created_at: number;
  image_base64: string;
}

export interface DeviceStreamProps {
  deviceId: string;
  frameData: FrameData | null;
  isConnected: boolean;
  onReconnect: () => void;
  fps: number;
  onFpsChange: (fps: number) => void;
}

export interface FrameBuffer {
  frames: FrameData[];
  maxSize: number;
  targetFps: number;
  lastPlayTime: number;
}