import { Router, Request, Response } from 'express';
import { playbackService } from '../../services/playbackService';

export const processingRouter = Router();

processingRouter.get('/status', (req: Request, res: Response) => {
  res.json(playbackService.getStatus());
});

processingRouter.post('/start', (req: Request, res: Response) => {
  const { target_fps = 10.0, mode = 'foveated' } = req.body;
  playbackService.start(Number(target_fps), mode);
  res.json({
    status: 'streaming',
    target_fps: Number(target_fps),
    mode,
  });
});

processingRouter.post('/pause', (req: Request, res: Response) => {
  playbackService.pause();
  res.json({
    status: 'paused',
    current_frame: playbackService.currentFrameIdx,
  });
});

processingRouter.post('/stop', (req: Request, res: Response) => {
  playbackService.stop();
  res.json({
    status: 'stopped',
    current_frame: 0,
  });
});

processingRouter.post('/seek', (req: Request, res: Response) => {
  const { frame_id = 0 } = req.body;
  const frameData = playbackService.seek(Number(frame_id));
  res.json({
    status: 'success',
    current_frame: playbackService.currentFrameIdx,
    frame_data: frameData,
  });
});
