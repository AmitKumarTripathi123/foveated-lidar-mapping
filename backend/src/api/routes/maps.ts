import { Router, Request, Response } from 'express';
import { playbackService } from '../../services/playbackService';

export const mapsRouter = Router();

mapsRouter.get('/current', (req: Request, res: Response) => {
  const framePayload = playbackService.processFrame(playbackService.currentFrameIdx);
  res.json(framePayload.map);
});

mapsRouter.get('/:frame_id', (req: Request, res: Response) => {
  const frameId = parseInt(req.params.frame_id, 10) || 0;
  const framePayload = playbackService.processFrame(frameId);
  res.json(framePayload.map);
});
