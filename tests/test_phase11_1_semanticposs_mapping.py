"""Tests for Phase 11.1 SemanticPOSS Authoritative Label Remapper."""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.semanticposs_label_mapping import (
    SemanticPOSSLabelRemapper,
    SemanticPOSSMappingError,
    VALID_SIH_CLASSES,
)


class TestPhase11_1SemanticPOSSMapping(unittest.TestCase):
    """Test suite for SemanticPOSS remapping rules, validation, and error policies."""

    @classmethod
    def setUpClass(cls):
        """Set up remapper config path."""
        cls.config_path = repo_root / "configs/semanticposs_label_mapping.yaml"

    def test_01_standard_poss_mapping(self):
        """Test 1: Standard POSS raw classes map to exact target SIH superclasses."""
        remapper = SemanticPOSSLabelRemapper()
        raw = np.array([22, 19, 20, 8, 9, 15, 17, 4, 7, 21, 0, 1], dtype=np.uint32)
        mapped = remapper.remap(raw)
        expected = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 255, 255], dtype=np.uint8)
        np.testing.assert_array_equal(mapped, expected)

    def test_02_unknown_label_error_policy(self):
        """Test 2: Unmapped raw IDs raise error when policy is 'error'."""
        remapper = SemanticPOSSLabelRemapper(unmapped_policy="error")
        bad_raw = np.array([9999], dtype=np.uint32)
        with self.assertRaises(SemanticPOSSMappingError):
            remapper.remap(bad_raw)

    def test_03_load_from_yaml(self):
        """Test 3: Initializing remapper from YAML config file."""
        remapper = SemanticPOSSLabelRemapper.from_yaml(self.config_path)
        raw = np.array([22, 15, 7, 0], dtype=np.uint32)
        mapped = remapper.remap(raw)
        self.assertEqual(mapped.tolist(), [0, 2, 3, 255])

    def test_04_audit_function(self):
        """Test 4: Audit returns accurate class distributions and percentages."""
        remapper = SemanticPOSSLabelRemapper()
        raw = np.array([22, 22, 15, 7, 0], dtype=np.uint32)
        audit = remapper.audit(raw)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["total_points"], 5)
        self.assertEqual(audit["sih_distribution"][0], 2)


if __name__ == "__main__":
    unittest.main()
