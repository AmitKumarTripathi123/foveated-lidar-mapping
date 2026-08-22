"""
Unit and Regression Tests for LiDAR Pipeline Visualizer & Debugger.
Verifies:
  1. PipelineVisualizer execution and intermediate state capture.
  2. PointTracer point lifecycle tracking.
  3. 64-bit integer packed voxelization coordinate encoding/decoding.
  4. Diagnostic plot generator file creation.
  5. Standalone interactive HTML dashboard generation.
  6. Debug artifact exporter (.npz, .json) structure and integrity.
"""

import unittest
import json
from pathlib import Path
import numpy as np

from visualization.pipeline_visualizer import PipelineVisualizer, PointTrace, PipelineIntermediateState
from visualization.plot_generator import PipelinePlotGenerator
from visualization.interactive_html import generate_interactive_html_dashboard
from visualization.debug_exporter import DebugArtifactExporter


class TestPipelineVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visualizer = PipelineVisualizer(device="cpu")
        cls.bin_path = "dataset/sequences/00/velodyne/000000.bin"
        cls.test_out_dir = Path("/tmp/test_vis_debug")
        cls.test_out_dir.mkdir(parents=True, exist_ok=True)

    def test_01_pipeline_visualizer_state(self):
        """Test 1: Visualizer runs pipeline and extracts valid intermediate state."""
        state = self.visualizer.process_scan(self.bin_path)
        self.assertIsInstance(state, PipelineIntermediateState)
        self.assertGreater(len(state.raw_points), 50000)
        self.assertGreater(len(state.foveated_points), 30000)
        self.assertGreater(state.unique_voxel_count, 20000)
        self.assertEqual(len(state.predicted_classes), len(state.foveated_points))
        self.assertGreater(state.grid_occupied_cells, 10000)
        self.assertGreater(state.total_pipeline_ms, 0.0)

    def test_02_point_lifecycle_tracer(self):
        """Test 2: Point tracer tracks a single point across all stages accurately."""
        state = self.visualizer.process_scan(self.bin_path)
        trace = self.visualizer.trace_point(state, point_idx=100)
        self.assertIsInstance(trace, PointTrace)
        self.assertEqual(trace.point_index, 100)
        self.assertEqual(len(trace.raw_xyz), 3)
        self.assertIn(trace.band_name, ["near_field", "mid_field", "mid_far_field", "far_field", "ultra_far"])
        self.assertEqual(trace.voxel_coords, trace.recovered_voxel_coords)
        self.assertTrue(0.0 <= trace.confidence <= 1.0)

    def test_03_packed_64bit_voxel_reversibility(self):
        """Test 3: 64-bit integer spatial hash encoding/decoding is 100% reversible."""
        state = self.visualizer.process_scan(self.bin_path)
        for s in state.packed_keys_sample:
            self.assertTrue(s["verified"])
            self.assertEqual(s["original_voxel"], s["recovered_voxel"])

    def test_04_plot_generation(self):
        """Test 4: PipelinePlotGenerator produces non-empty PNG files for all stages."""
        state = self.visualizer.process_scan(self.bin_path)
        trace = self.visualizer.trace_point(state, point_idx=100)
        plot_gen = PipelinePlotGenerator(output_dir=self.test_out_dir / "plots")
        plots = plot_gen.generate_all_stage_plots(state, trace)

        for name, p in plots.items():
            self.assertTrue(p.exists(), f"Plot {name} does not exist at {p}")
            self.assertGreater(p.stat().st_size, 1000, f"Plot {name} file is too small")

    def test_05_interactive_html_dashboard(self):
        """Test 5: Standalone HTML dashboard is generated with valid structure."""
        state = self.visualizer.process_scan(self.bin_path)
        trace = self.visualizer.trace_point(state, point_idx=100)
        html_p = generate_interactive_html_dashboard(state, output_path=self.test_out_dir / "dashboard.html", trace=trace)

        self.assertTrue(html_p.exists())
        content = html_p.read_text(encoding="utf-8")
        self.assertIn("Foveated 2.5D LiDAR Pipeline Visualizer", content)
        self.assertIn("canvas", content)
        self.assertIn("pointsData", content)

    def test_06_debug_exporter(self):
        """Test 6: DebugArtifactExporter produces valid .npz and .json artifacts."""
        state = self.visualizer.process_scan(self.bin_path)
        trace = self.visualizer.trace_point(state, point_idx=100)
        exporter = DebugArtifactExporter(output_dir=self.test_out_dir)
        debug_files = exporter.export_all(state, trace)

        self.assertTrue(debug_files["npz"].exists())
        self.assertTrue(debug_files["json"].exists())

        with open(debug_files["json"], "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("scan", data)
            self.assertIn("raw_points", data)
            self.assertIn("spvcnn_latency_ms", data)
            self.assertIn("accuracy", data)


if __name__ == "__main__":
    unittest.main()
