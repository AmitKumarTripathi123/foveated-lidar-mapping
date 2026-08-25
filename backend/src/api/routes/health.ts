import { Router, Request, Response } from 'express';
import os from 'os';
import { CONFIG } from '../../config';

export const healthRouter = Router();

healthRouter.get('/', (req: Request, res: Response) => {
  const freeMemGb = Number((os.freemem() / (1024 ** 3)).toFixed(2));
  const totalMemGb = Number((os.totalmem() / (1024 ** 3)).toFixed(2));

  res.json({
    status: 'healthy',
    app_name: CONFIG.appName,
    version: CONFIG.version,
    runtime: `Node.js ${process.version}`,
    platform: `${os.type()} ${os.release()} (${os.arch()})`,
    cpu_count: os.cpus().length,
    memory_total_gb: totalMemGb,
    memory_available_gb: freeMemGb,
    cuda_available: false,
    environment: '100% Pure JavaScript / TypeScript Ecosystem',
  });
});
