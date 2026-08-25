import { Router, Request, Response } from 'express';
import { CONFIG } from '../../config';

export const configRouter = Router();

configRouter.get('/', (req: Request, res: Response) => {
  res.json({
    app_name: CONFIG.appName,
    version: CONFIG.version,
    zones: CONFIG.zones,
    default_fps: CONFIG.defaultFps,
    max_fps: CONFIG.maxFps,
  });
});
