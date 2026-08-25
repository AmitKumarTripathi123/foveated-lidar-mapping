import {
  RawPoint,
  SemanticPoint,
  BoundingBox3D,
  FoveatedCell,
  BenchmarkComparison,
  DatasetInfo,
  FoveatedZoneConfig,
} from '../types/lidar';

export interface IDatasetAdapter {
  listSequences(): Promise<DatasetInfo[]> | DatasetInfo[];
  loadSequence(sequenceId: string): Promise<number> | number;
  getFrame(frameIdx: number): Promise<RawPoint[]> | RawPoint[];
}

export interface IAIModelAdapter {
  predictSemantics(
    rawPoints: RawPoint[],
    frameIdx: number
  ): Promise<{
    semanticPoints: SemanticPoint[];
    boundingBoxes: BoundingBox3D[];
    aiLatencyMs: number;
  }> | {
    semanticPoints: SemanticPoint[];
    boundingBoxes: BoundingBox3D[];
    aiLatencyMs: number;
  };
}

export interface IGridEngineAdapter {
  computeFoveatedGrid(
    semanticPoints: SemanticPoint[],
    zonesConfig?: FoveatedZoneConfig[]
  ): Promise<{
    cells: FoveatedCell[];
    gridLatencyMs: number;
  }> | {
    cells: FoveatedCell[];
    gridLatencyMs: number;
  };
}

export interface IBenchmarkAdapter {
  computeComparison(
    frameId: number,
    rawPointCount: number,
    foveatedCells: FoveatedCell[],
    gridLatencyMs: number
  ): BenchmarkComparison;
}
