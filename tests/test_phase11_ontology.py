"""Tests for Phase 11 Authoritative Label Ontology and Remapping."""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_labels
from ml.data.authoritative_label_mapping import (
    AuthoritativeLabelRemapper,
    AuthoritativeMappingError,
    VALID_SIH_CLASSES,
)


class TestPhase11Ontology(unittest.TestCase):
    """Test suite for ontology verification, mapping, and unknown label handling."""

    @classmethod
    def setUpClass(cls):
        """Set up class paths."""
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"
        cls.config_path = repo_root / "configs/authoritative_label_mapping.yaml"

    def test_01_dataset_identity_validation(self):
        """Test 1: Real frame contains SemanticKITTI label IDs (40, 48, 50, 70, etc)."""
        lbls = load_labels(self.lbl_file)
        unique_raw = set(np.unique(lbls))
        self.assertIn(40, unique_raw)  # road
        self.assertIn(10, unique_raw)  # car
        self.assertIn(70, unique_raw)  # vegetation

    def test_02_raw_label_enumeration(self):
        """Test 2: Authoritative remapper audit returns complete distribution."""
        lbls = load_labels(self.lbl_file)
        remapper = AuthoritativeLabelRemapper(dataset_name="SemanticKITTI")
        audit = remapper.audit(lbls)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["total_points"], 66658)

    def test_03_authoritative_remapping(self):
        """Test 3: Authoritative remapper maps all raw labels to {0, 1, 2, 3, 255}."""
        lbls = load_labels(self.lbl_file)
        remapper = AuthoritativeLabelRemapper(dataset_name="SemanticKITTI")
        mapped = remapper.remap(lbls)
        self.assertTrue(set(np.unique(mapped)).issubset(VALID_SIH_CLASSES))

    def test_04_semanticposs_remapping(self):
        """Test 4: Authoritative remapper correctly maps SemanticPOSS labels."""
        poss_raw = np.array([22, 19, 15, 7, 0], dtype=np.uint32)
        remapper = AuthoritativeLabelRemapper(dataset_name="SemanticPOSS")
        mapped = remapper.remap(poss_raw)
        np.testing.assert_array_equal(mapped, np.array([0, 1, 2, 3, 255], dtype=np.uint8))

    def test_05_unknown_label_error_policy(self):
        """Test 5: Remapper raises AuthoritativeMappingError when policy is 'error' and unknown labels occur."""
        remapper = AuthoritativeLabelRemapper(dataset_name="SemanticKITTI", unmapped_policy="error")
        unknown_labels = np.array([9999], dtype=np.uint32)
        with self.assertRaises(AuthoritativeMappingError):
            remapper.remap(unknown_labels)

    def test_06_load_from_yaml(self):
        """Test 6: AuthoritativeLabelRemapper initializes cleanly from YAML config."""
        remapper = AuthoritativeLabelRemapper.from_yaml(self.config_path)
        self.assertEqual(remapper.dataset_name, "SemanticKITTI")


if __name__ == "__main__":
    unittest.main()
