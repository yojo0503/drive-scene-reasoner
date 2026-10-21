import importlib.util
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location("extract_dataset", Path(__file__).parents[1] / "scripts/extract_dataset.py")
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def test_zip_extraction_and_resume(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("nuscenes/samples/CAM_FRONT/image.jpg", b"synthetic format fixture")
    out = tmp_path / "out"
    assert extractor.extract(archive, out)["files"] == 1
    image = out / "nuscenes/samples/CAM_FRONT/image.jpg"
    image.write_bytes(b"x" * len(b"synthetic format fixture"))
    extractor.extract(archive, out)
    assert image.read_bytes() == b"synthetic format fixture"


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "foo\\bar"])
def test_zip_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        extractor.safe_target(tmp_path, name)


def test_parallel_ranges_resume_prefix_and_assemble_exactly(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("download_assets", Path(__file__).parents[1] / "scripts/download_assets.py")
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    expected = b"0123456789abcdef"
    partial = tmp_path / "weights.partial"
    partial.write_bytes(expected[:6])

    def fake_curl(command, text):
        assert "--continue-at" not in command
        start, end = map(int, command[command.index("--range") + 1].split("-"))
        target = Path(command[command.index("--output") + 1])
        target.write_bytes(expected[start:end + 1])
        return "206"

    monkeypatch.setattr(downloader.subprocess, "check_output", fake_curl)
    downloader.ranged_download(["curl", "--continue-at", "-", "--output", str(partial)], "unused", partial, len(expected), 4)
    assert partial.read_bytes() == expected


def test_parallel_ranges_reject_a_server_ignoring_range(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("download_assets", Path(__file__).parents[1] / "scripts/download_assets.py")
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)

    def fake_curl(command, text):
        Path(command[command.index("--output") + 1]).write_bytes(b"1234")
        return "200"

    monkeypatch.setattr(downloader.subprocess, "check_output", fake_curl)
    with pytest.raises(ValueError, match="exact requested range"):
        downloader.ranged_download(["curl"], "unused", tmp_path / "bad.partial", 8, 2)
