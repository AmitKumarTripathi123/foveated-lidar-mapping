import { Router, Request, Response } from 'express';
import { playbackService } from '../../services/playbackService';

export const metricsRouter = Router();

metricsRouter.get('/current', (req: Request, res: Response) => {
  const framePayload = playbackService.processFrame(playbackService.currentFrameIdx);
  res.json(framePayload.metrics);
});
