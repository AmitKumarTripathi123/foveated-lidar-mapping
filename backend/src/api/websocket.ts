import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';
import { playbackService } from '../services/playbackService';

export function setupWebSocketServer(httpServer: Server) {
  const wss = new WebSocketServer({ server: httpServer, path: '/ws/stream' });

  wss.on('connection', (ws: WebSocket) => {
    console.log('[WebSocket] Client connected to /ws/stream');
    playbackService.subscribe(ws);

    // Send initial frame immediately upon connection
    try {
      const initialFrame = playbackService.processFrame(playbackService.currentFrameIdx);
      ws.send(JSON.stringify(initialFrame));
    } catch (err) {
      console.error('[WebSocket] Error sending initial frame:', err);
    }

    ws.on('message', (message: string) => {
      try {
        const msg = JSON.parse(message.toString());
        const { action, payload = {} } = msg;

        if (action === 'play') {
          const fps = payload.fps || 10.0;
          const mode = payload.mode || 'foveated';
          playbackService.start(fps, mode);
        } else if (action === 'pause') {
          playbackService.pause();
        } else if (action === 'stop') {
          playbackService.stop();
        } else if (action === 'seek') {
          const frameId = payload.frame_id ?? 0;
          playbackService.seek(frameId);
        } else if (action === 'set_fps') {
          playbackService.setFps(payload.fps || 10.0);
        }
      } catch (err) {
        console.error('[WebSocket] Invalid message JSON:', err);
      }
    });

    ws.on('close', () => {
      console.log('[WebSocket] Client disconnected.');
      playbackService.unsubscribe(ws);
    });

    ws.on('error', (err) => {
      console.error('[WebSocket] Connection error:', err);
      playbackService.unsubscribe(ws);
    });
  });

  return wss;
}
