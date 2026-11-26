"""
tests/test_main_integration.py
================================
Integration tests for main.py CLI execution across image and video modes.
"""

import sys
import os
import json
import subprocess
import pytest
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


def create_temp_image(path: str, brightness: int = 150):
    img = np.full((240, 320, 3), brightness, dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def create_temp_video(path: str, num_frames: int = 5):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 20.0, (320, 240))
    for i in range(num_frames):
        frame = np.full((240, 320, 3), 100 + i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


class TestMainApplicationCLI:

    def test_main_image_mode_execution(self, tmp_path):
        """Test main application runs end-to-end in image mode."""
        img_path = str(tmp_path / "test_input.jpg")
        create_temp_image(img_path)
        out_dir = str(tmp_path / "out_img")

        test_args = [
            "main.py",
            "--mode", "image",
            "--input", img_path,
            "--output", out_dir,
            "--save-params",
        ]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = test_args
            exit_code = main.main()
            assert exit_code == 0
        finally:
            sys.argv = old_argv

        # Check outputs
        assert os.path.isfile(os.path.join(out_dir, "test_input_annotated.jpg"))
        assert os.path.isfile(os.path.join(out_dir, "test_input_detections.json"))
        assert os.path.isfile(os.path.join(out_dir, "test_input_scene.json"))
        assert os.path.isfile(os.path.join(out_dir, "parameter_report.json"))

        with open(os.path.join(out_dir, "test_input_detections.json")) as f:
            dets = json.load(f)
            assert "detections" in dets

        with open(os.path.join(out_dir, "test_input_scene.json")) as f:
            scene = json.load(f)
            assert "raw_condition" in scene
            assert "smoothed_condition" in scene

    def test_main_video_mode_execution(self, tmp_path):
        """Test main application runs end-to-end in video mode."""
        vid_path = str(tmp_path / "test_input.mp4")
        create_temp_video(vid_path, num_frames=6)
        out_dir = str(tmp_path / "out_vid")

        test_args = [
            "main.py",
            "--mode", "video",
            "--input", vid_path,
            "--output", out_dir,
            "--no-display",
            "--max-frames", "4",
        ]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = test_args
            exit_code = main.main()
            assert exit_code == 0
        finally:
            sys.argv = old_argv

        assert os.path.isfile(os.path.join(out_dir, "test_input_ratfa.mp4"))
        assert os.path.isfile(os.path.join(out_dir, "test_input_detections.json"))
        assert os.path.isfile(os.path.join(out_dir, "test_input_scene.json"))

        with open(os.path.join(out_dir, "test_input_detections.json")) as f:
            data = json.load(f)
            assert data["total_frames"] == 4

    def test_main_invalid_input_path(self, tmp_path):
        """Test main handles missing input path gracefully."""
        test_args = [
            "main.py",
            "--mode", "image",
            "--input", str(tmp_path / "nonexistent.jpg"),
        ]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = test_args
            exit_code = main.main()
            assert exit_code != 0
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

