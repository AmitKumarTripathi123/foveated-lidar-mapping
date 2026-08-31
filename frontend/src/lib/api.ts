import { API_BASE_URL } from './constants';
import { DatasetInfo, BenchmarkComparison, SystemMetrics, FoveatedCell } from '@/types/lidar';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchDatasets(): Promise<{ datasets: DatasetInfo[]; active_dataset: string }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function loadDataset(datasetId: string) {
  const res = await fetch(`${API_BASE_URL}/api/v1/datasets/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  if (!res.ok) throw new Error('Failed to load dataset sequence');
  return res.json();
}

export const switchDataset = loadDataset;

export async function startPlayback(targetFps: number = 10, mode: string = 'foveated') {
  const res = await fetch(`${API_BASE_URL}/api/v1/processing/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_fps: targetFps, mode }),
  });
  return res.json();
}

export async function pausePlayback() {
  const res = await fetch(`${API_BASE_URL}/api/v1/processing/pause`, {
    method: 'POST',
  });
  return res.json();
}

export async function stopPlayback() {
  const res = await fetch(`${API_BASE_URL}/api/v1/processing/stop`, {
    method: 'POST',
  });
  return res.json();
}

export async function seekFrame(frameId: number) {
  const res = await fetch(`${API_BASE_URL}/api/v1/processing/seek`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame_id: frameId }),
  });
  return res.json();
}

export async function fetchBenchmark(frameId?: number): Promise<BenchmarkComparison> {
  const url = frameId !== undefined 
    ? `${API_BASE_URL}/api/v1/benchmark/compare?frame_id=${frameId}`
    : `${API_BASE_URL}/api/v1/benchmark/compare`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch benchmark comparison');
  return res.json();
}
