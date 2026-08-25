import { FoveatedZoneConfig } from './types/lidar';
import path from 'path';

export const REPO_ROOT = path.resolve(__dirname, '../../amit_repo');

export const CONFIG = {
  appName: 'SIH 3D LiDAR Foveated Mapping Platform',
  repoName: 'AmitKumarTripathi123/foveated-lidar-mapping',
  repoUrl: 'https://github.com/AmitKumarTripathi123/foveated-lidar-mapping',
  version: '1.0.0-freeze',
  certificationStage: 'Phase 17 AI/ML Production Freeze',
  modelArchitecture: 'SPVCNN (Sparse Point-Voxel Convolution)',
  modelParameters: 138514,
  validationMIoU: 53.59,
  meanSequenceMIoU: 51.94,
  dynamicObjectIoU: 43.68,
  sensorType: 'Hesai Pandar40 (40-Beam LiDAR, 10 Hz, 100m Range)',
  port: parseInt(process.env.PORT || '8000', 10),
  host: '0.0.0.0',
  corsOrigins: ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8000'],
  defaultFps: 10.0,
  maxFps: 30.0,
  zones: [
    {
      zone_id: 0,
      name: 'Zone 0 — Near-Field (Foveal)',
      radius_min: 0.0,
      radius_max: 10.0,
      resolution: 0.05,
      description: '5 cm resolution for precision curb & immediate obstacle avoidance',
    },
    {
      zone_id: 1,
      name: 'Zone 1 — Mid-Field (Intermediate)',
      radius_min: 10.0,
      radius_max: 40.0,
      resolution: 0.15,
      description: '15 cm resolution for lane geometry & dynamic vehicle/pedestrian tracking',
    },
    {
      zone_id: 2,
      name: 'Zone 2 — Far-Field (Peripheral)',
      radius_min: 40.0,
      radius_max: 100.0,
      resolution: 0.50,
      description: '50 cm resolution for macro road corridor & boundary tracking (~85% memory savings)',
    },
  ] as FoveatedZoneConfig[],
};
