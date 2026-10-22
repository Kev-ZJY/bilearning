#!/usr/bin/env python3
"""Build a macOS development app. No private data or runtime is copied into it."""

import argparse
import json
from pathlib import Path
import plistlib
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--python", default=str(Path("backend/.venv/bin/python").absolute())
)
parser.add_argument("--data", help="Optional isolated test app-data directory")
parser.add_argument("--output", default="desktop/build/视频学习工作台.app")
parser.add_argument(
    "--bundle-id",
    default="org.video-workbench.desktop",
    help="Separate identifier for isolated UI acceptance",
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
python = Path(args.python).expanduser().absolute()
if not python.exists():
    parser.error("Specify an installed backend Python with --python")
subprocess.run(
    [str(python), "-c", "import fastapi,uvicorn,sqlalchemy,faster_whisper,reportlab"],
    check=True,
)
subprocess.run(["npm", "run", "build"], cwd=root / "frontend", check=True)
app = Path(args.output).absolute()
macos = app / "Contents/MacOS"
resources = app / "Contents/Resources"
macos.mkdir(parents=True, exist_ok=True)
resources.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "xcrun",
        "swiftc",
        str(root / "desktop/main.swift"),
        str(root / "desktop/NavigationPolicy.swift"),
        "-o",
        str(macos / "VideoWorkbench"),
        "-framework",
        "AppKit",
        "-framework",
        "WebKit",
        "-framework",
        "Security",
        "-module-cache-path",
        "/tmp/video-workbench-swift-cache",
    ],
    check=True,
)
config = {"root": str(root), "python": str(python)}
if args.data:
    config["data"] = str(Path(args.data).expanduser().absolute())
(resources / "development.json").write_text(json.dumps(config))
(app / "Contents/Info.plist").write_bytes(
    plistlib.dumps(
        {
            "CFBundleName": "视频学习工作台",
            "CFBundleDisplayName": "视频学习工作台",
            "CFBundleIdentifier": args.bundle_id,
            "CFBundleExecutable": "VideoWorkbench",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
            "NSHumanReadableCopyright": "Independent learning tool; not affiliated with Bilibili.",
        }
    )
)
print(app)
