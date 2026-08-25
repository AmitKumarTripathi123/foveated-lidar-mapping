import { Router, Request, Response } from 'express';
import { playbackService } from '../../services/playbackService';

export const benchmarksRouter = Router();

benchmarksRouter.get('/compare', (req: Request, res: Response) => {
  const frameId = req.query.frame_id ? parseInt(req.query.frame_id as string, 10) : playbackService.currentFrameIdx;
  const framePayload = playbackService.processFrame(frameId);
  res.json(framePayload.benchmark);
});
