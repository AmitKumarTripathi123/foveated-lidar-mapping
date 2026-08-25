export interface RawPoint {
  x: number;
  y: number;
  z: number;
  intensity: number;
}

export interface SemanticPoint {
  x: number;
  y: number;
  z: number;
  intensity: number;
  semantic_class: number;
  class_name: string;
  confidence: number;
}

export interface BoundingBox3D {
  id: string;
  class_name: string;
  confidence: number;
  center: [number, number, number];
  size: [number, number, number];
  rotation_yaw: number;
}

export interface FrameMetadata {
  frame_id: number;
  timestamp_ms: number;
  total_points: number;
  sequence_id: string;
  bounding_boxes: BoundingBox3D[];
}

export interface FoveatedCell {
  x: number;
  y: number;
  elevation: number;
  resolution: number;
  zone_id: number;
  semantic_class: number;
  class_name: string;
  confidence: number;
  point_count: number;
  traversability: number;
  roughness: number;
}

export interface FoveatedMapFrame {
  frame_id: number;
  timestamp_ms: number;
  total_cells: number;
  zone_distribution: Record<number, number>;
  cells: FoveatedCell[];
}

export interface SystemMetrics {
  fps: number;
  total_latency_ms: number;
  ai_latency_ms: number;
  grid_latency_ms: number;
  memory_ram_mb: number;
  memory_vram_mb: number;
  cpu_percent: number;
  raw_point_count: number;
  cell_count: number;
  compression_ratio_percent: number;
}

export interface UniformGridMetrics {
  resolution_m: number;
  cell_count: number;
  memory_mb: number;
  processing_latency_ms: number;
  fps: number;
}

export interface FoveatedGridMetrics {
  near_resolution_m: number;
  far_resolution_m: number;
  cell_count: number;
  memory_mb: number;
  processing_latency_ms: number;
  fps: number;
  memory_savings_percent: number;
  speedup_factor: number;
}

export interface BenchmarkComparison {
  frame_id: number;
  uniform: UniformGridMetrics;
  foveated: FoveatedGridMetrics;
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  total_frames: number;
  fps: number;
  point_format: string;
  sensor_type: string;
}

export interface FoveatedZoneConfig {
  zone_id: number;
  name: string;
  radius_min: number;
  radius_max: number;
  resolution: number;
  description: string;
}
