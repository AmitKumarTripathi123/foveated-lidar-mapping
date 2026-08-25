import { Router, Request, Response } from 'express';
import { playbackService } from '../../services/playbackService';

export const datasetsRouter = Router();

datasetsRouter.get('/', (req: Request, res: Response) => {
  const datasets = playbackService.datasetAdapter.listSequences();
  res.json({ datasets });
});

datasetsRouter.post('/load', (req: Request, res: Response) => {
  const { dataset_id } = req.body;
  if (!dataset_id) {
    return res.status(400).json({ error: 'dataset_id is required' });
  }
  const totalFrames = playbackService.loadSequence(dataset_id);
  res.json({
    dataset_id,
    total_frames: totalFrames,
    status: 'ready',
  });
});
