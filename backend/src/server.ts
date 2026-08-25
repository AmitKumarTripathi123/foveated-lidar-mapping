import express, { Request, Response } from 'express';
import http from 'http';
import cors from 'cors';
import { CONFIG } from './config';
import { healthRouter } from './api/routes/health';
import { datasetsRouter } from './api/routes/datasets';
import { processingRouter } from './api/routes/processing';
import { mapsRouter } from './api/routes/maps';
import { metricsRouter } from './api/routes/metrics';
import { benchmarksRouter } from './api/routes/benchmarks';
import { configRouter } from './api/routes/config';
import { setupWebSocketServer } from './api/websocket';
import { playbackService } from './services/playbackService';

const app = express();

// Middleware
app.use(cors({ origin: '*' })); // Allow all origins for hackathon development
app.use(express.json());

// API Routes
app.use('/api/v1/health', healthRouter);
app.use('/api/v1/datasets', datasetsRouter);
app.use('/api/v1/processing', processingRouter);
app.use('/api/v1/map', mapsRouter);
app.use('/api/v1/metrics', metricsRouter);
app.use('/api/v1/benchmark', benchmarksRouter);
app.use('/api/v1/config', configRouter);

// Root Info Route
app.get('/', (req: Request, res: Response) => {
  res.json({
    project: 'SIH 2026: 3D LiDAR Perception & Foveated 2.5D Mapping Platform',
    developer: 'AYUSH (Lead Full-Stack & Systems Integration Engineer)',
    environment: '100% Pure JavaScript / TypeScript Ecosystem (Node.js + Express + WebSocket)',
    rest_api: '/api/v1',
    ws_stream: 'ws://localhost:8000/ws/stream',
    health: '/api/v1/health',
  });
});

// Create HTTP server
const server = http.createServer(app);

// Setup WebSocket server on same HTTP server instance
setupWebSocketServer(server);

// Start server
server.listen(CONFIG.port, CONFIG.host, () => {
  console.log(`=======================================================`);
  console.log(`🚀 ${CONFIG.appName}`);
  console.log(`🌐 Server running at: http://localhost:${CONFIG.port}`);
  console.log(`📡 WebSocket stream: ws://localhost:${CONFIG.port}/ws/stream`);
  console.log(`💻 Environment: 100% Pure JavaScript/TypeScript (Node.js)`);
  console.log(`=======================================================`);

  // Start continuous 10Hz streaming by default so data is always streaming
  playbackService.start(10.0, 'foveated');
});
