/**
 * Single Authoritative Source of Truth for Scientific Metrics and Benchmarks
 * Project: Adaptive Variable-Resolution 2.5D LiDAR Mapping
 * Track: Smart India Hackathon (SIH)
 */

export const SCIENTIFIC_BENCHMARKS = {
  // 1. Controlled Reference Benchmark Frame
  referenceFrameId: 0,
  referenceInputPoints: 1248531,
  referenceUniformOccupiedCells: 45820,
  referenceFoveatedOccupiedCells: 9169,

  // 2. Theoretical Grid Address Space Capacities (Analytical Ring Integration)
  theoreticalUniformCapacity: 12566370,
  theoreticalFoveatedCapacity: 340549,
  theoreticalCapacityReductionPercent: 97.29,

  // Zone Capacities
  zone0TheoreticalCapacity: 125664,  // pi * 10^2 / 0.05^2
  zone1TheoreticalCapacity: 120637,  // pi * (50^2 - 10^2) / 0.25^2
  zone2TheoreticalCapacity: 94248,   // pi * (100^2 - 50^2) / 0.50^2

  // 3. Reference Latencies
  referenceAiLatencyMs: 18.2, // SPVCNN backbone inference on NVIDIA RTX 3070
  referenceUniformGridLatencyMs: 55.6,
  referenceFoveatedGridLatencyMs: 12.1,
  referenceUniformTotalPipelineLatencyMs: 73.8,
  referenceFoveatedTotalPipelineLatencyMs: 30.3,

  // 4. Sensor & Foveation Spatial Specifications
  sensorSpinRateHz: 10.0, // Physical LiDAR spin frequency
  zone0RadiusMaxM: 10.0,
  zone0ResolutionM: 0.05,
  zone1RadiusMaxM: 50.0,
  zone1ResolutionM: 0.25,
  zone2RadiusMaxM: 100.0,
  zone2ResolutionM: 0.50,

  // 5. Memory Struct Model (64 Bytes / cell footprint)
  bytesPerCellStruct: 64,
  theoreticalUniformBufferMb: 785.4,
  theoreticalFoveatedBufferMb: 21.8,
  benchmarkUniformBufferMb: 2.93,
  benchmarkFoveatedBufferMb: 0.59,
  estimatedHostRamMb: 134.8,

  // 6. Engineering Criteria & Targets
  targetPipelineRateHz: 30.0, // Project-defined engineering real-time threshold
  displayTargetFps: 60,
} as const;

export interface MetricProvenanceEntry {
  metric: string;
  value: string;
  classification:
    | 'MEASURED (LIVE)'
    | 'REFERENCE BENCHMARK'
    | 'THEORETICAL'
    | 'ESTIMATED'
    | 'DERIVED'
    | 'SPECIFICATION';
  provenance: string;
}

export const METRIC_PROVENANCE_TABLE: MetricProvenanceEntry[] = [
  {
    metric: 'Zone 0 Resolution',
    value: '0.05 m',
    classification: 'SPECIFICATION',
    provenance: 'r \u2264 10 m (Near-field ego vehicle corridor)',
  },
  {
    metric: 'Zone 1 Resolution',
    value: '0.25 m',
    classification: 'SPECIFICATION',
    provenance: '10 < r \u2264 50 m (Intermediate roadway)',
  },
  {
    metric: 'Zone 2 Resolution',
    value: '0.50 m',
    classification: 'SPECIFICATION',
    provenance: '50 < r \u2264 100 m (Far-field perimeter)',
  },
  {
    metric: 'Foveated Capacity',
    value: '340,549 cells',
    classification: 'THEORETICAL',
    provenance: 'Analytical annular-cell sum: \u03c0(10\u00b2)/0.05\u00b2 + \u03c0(50\u00b2-10\u00b2)/0.25\u00b2 + \u03c0(100\u00b2-50\u00b2)/0.50\u00b2',
  },
  {
    metric: 'Uniform Capacity',
    value: '12,566,370 cells',
    classification: 'THEORETICAL',
    provenance: 'Uniform 5 cm circle: \u03c0(100\u00b2)/0.05\u00b2',
  },
  {
    metric: 'Capacity Reduction',
    value: '97.29%',
    classification: 'DERIVED',
    provenance: 'Analytical reduction: (12,566,370 - 340,549) / 12,566,370 \u00d7 100',
  },
  {
    metric: 'Foveated Benchmark Cells',
    value: '9,169 cells',
    classification: 'REFERENCE BENCHMARK',
    provenance: 'Controlled reference frame 0 benchmark run',
  },
  {
    metric: 'Uniform Benchmark Cells',
    value: '45,820 cells',
    classification: 'REFERENCE BENCHMARK',
    provenance: 'Controlled reference frame 0 benchmark run (uniform 5 cm)',
  },
  {
    metric: 'Grid Generation Latency',
    value: '12.1\u201313.8 ms',
    classification: 'MEASURED (LIVE)',
    provenance: 'Browser/client performance.now() timer across 2.5D binning loop',
  },
  {
    metric: 'AI Inference Latency',
    value: '18.2 ms',
    classification: 'REFERENCE BENCHMARK',
    provenance: 'SPVCNN 3D neural net inference on NVIDIA RTX 3070',
  },
  {
    metric: 'AI + Grid Latency',
    value: '30.3\u201332.0 ms',
    classification: 'DERIVED',
    provenance: 'Hybrid sum: 18.2 ms (AI Ref) + Live Grid Generation Latency',
  },
  {
    metric: 'Pipeline Throughput',
    value: '31.3\u201333.0 Hz',
    classification: 'DERIVED',
    provenance: 'Calculated as 1000 / (AI + Grid Pipeline Latency); meets 30 Hz target',
  },
  {
    metric: 'LiDAR Spin Rate',
    value: '10.0 Hz',
    classification: 'SPECIFICATION',
    provenance: 'Standard physical LiDAR sensor acquisition frequency (100 ms period)',
  },
  {
    metric: 'Buffer Footprint',
    value: '21.8 MB vs 785.4 MB',
    classification: 'ESTIMATED',
    provenance: '64 B/cell struct model: 340,549 \u00d7 64 B vs 12,566,370 \u00d7 64 B',
  },
  {
    metric: 'Host Process RAM',
    value: '134.8 MB',
    classification: 'ESTIMATED',
    provenance: 'Node.js process heap memory baseline',
  },
  {
    metric: 'WebGL FPS',
    value: '~60 FPS',
    classification: 'MEASURED (LIVE)',
    provenance: 'Live Three.js useFrame / requestAnimationFrame browser measurement',
  },
];
