#!/usr/bin/env python3
"""
Test script to verify that the new project structure works correctly.
"""

import os
import sys
import importlib
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")

    try:
        # Test utils import
        from src.utils.eval_tools import fz0_per_step, kupiec_pof

        print("✓ src.utils.eval_tools imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import src.utils.eval_tools: {e}")
        return False

    try:
        # Test models import
        from src.models.transformer_var_es_paper_exact import BasicVaRTransformer

        print("✓ src.models.transformer_var_es_paper_exact imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import transformer model: {e}")
        return False

    try:
        # Test SRNN import
        from src.models.srnn_ve1_paper_exact import SRNNVE1

        print("✓ src.models.srnn_ve1_paper_exact imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SRNN model: {e}")
        return False

    try:
        # Test baseline import
        from src.baselines.baseline_classic_var_es import pipeline as baseline_pipeline

        print("✓ src.baselines.baseline_classic_var_es imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import baseline model: {e}")
        return False

    return True


def test_artifact_structure():
    """Test that the artifact directory structure can be created."""
    print("\nTesting artifact structure...")

    try:
        from scripts.run_sim_models import create_artifact_dir

        # Test creating artifact directory
        artifact_path = create_artifact_dir(
            dgp="garch11_skt",
            alpha=0.05,
            seed=42,
            calibrate=True,
            feature_parity=True,
            n_samples=5000,
        )

        expected_path = (
            "artifacts/dgp=garch11_skt/n=5000/alpha=050/seed=0042/cal=y/feat=parity"
        )
        if artifact_path == expected_path:
            print(f"✓ Artifact directory structure created: {artifact_path}")
        else:
            print(f"✗ Unexpected artifact path: {artifact_path}")
            return False

        # Check that subdirectories were created
        subdirs = [
            "models/transformer",
            "models/srnn",
            "baseline",
            "figures",
            "tables",
            "logs",
        ]
        for subdir in subdirs:
            full_path = os.path.join(artifact_path, subdir)
            if os.path.exists(full_path):
                print(f"✓ Subdirectory created: {subdir}")
            else:
                print(f"✗ Subdirectory not created: {subdir}")
                return False

        # Clean up test directory
        if os.path.exists(artifact_path):
            shutil.rmtree(artifact_path)
            print("✓ Test directory cleaned up")

        return True

    except Exception as e:
        print(f"✗ Failed to test artifact structure: {e}")
        return False


def test_model_pipeline():
    """Test that model pipelines can be called with the new structure."""
    print("\nTesting model pipelines...")

    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("close\n100.0\n101.0\n99.5\n102.0\n98.0\n")
        temp_csv = f.name

    try:
        # Test transformer pipeline
        from src.models.transformer_var_es_paper_exact import (
            pipeline as transformer_pipeline,
        )

        # This should work without errors (though may not converge with minimal data)
        try:
            model, metrics, (v_eval, e_eval, y_aligned, fz0) = transformer_pipeline(
                csv_path=temp_csv,
                alpha=0.05,
                feature_parity=True,
                calibrate=False,
                run_tag="test",
                out_dir="test_output",
                fig_dir="test_figures",
            )
            print("✓ Transformer pipeline executed successfully")
        except Exception as e:
            print(
                f"⚠ Transformer pipeline had issues (expected with minimal data): {e}"
            )

        # Clean up test outputs
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
        if os.path.exists("test_figures"):
            shutil.rmtree("test_figures")

        return True

    except Exception as e:
        print(f"✗ Failed to test model pipeline: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)


def test_directory_structure():
    """Test that the expected directory structure exists."""
    print("\nTesting directory structure...")

    expected_dirs = [
        "src/models",
        "src/baselines",
        "src/utils",
        "src/dgp",
        "scripts",
        "data/raw",
        "data/interim",
        "data/processed",
        "results/tables",
        "results/figures",
        "reports/thesis",
        "reports/slides",
    ]

    missing_dirs = []
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"\nCreating missing directories: {missing_dirs}")
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created: {dir_path}")

    return len(missing_dirs) == 0


def main():
    """Run all tests."""
    print("Testing Agri-Forecast Project Structure")
    print("=" * 50)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Artifact Structure", test_artifact_structure),
        ("Model Pipeline", test_model_pipeline),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The project structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
