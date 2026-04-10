"""
File operations tool — all writes are sandboxed to the output/ folder.
"""

import os
import pathlib

# Safety: all file operations are restricted to this directory
OUTPUT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_path(filename: str) -> pathlib.Path:
    """
    Resolve a safe path inside OUTPUT_DIR.
    Strips any path traversal attempts (e.g., ../../etc/passwd).
    """
    safe = OUTPUT_DIR / pathlib.Path(filename).name
    return safe


def create_file(filename: str, content: str = "") -> dict:
    """
    Create a new file in the output/ directory.
    Returns a result dict with status, path, and message.
    """
    try:
        path = safe_path(filename)

        # If file exists, add a numeric suffix
        if path.exists():
            stem = path.stem
            suffix = path.suffix
            counter = 1
            while path.exists():
                path = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "path": str(path),
            "filename": path.name,
            "message": f"Created `{path.name}` in output/",
            "content": content,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_folder(folder_name: str) -> dict:
    """Create a folder inside output/."""
    try:
        folder = OUTPUT_DIR / pathlib.Path(folder_name).name
        folder.mkdir(parents=True, exist_ok=True)
        return {
            "status": "success",
            "path": str(folder),
            "filename": folder.name,
            "message": f"Created folder `{folder.name}` in output/",
            "content": "",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_output_files() -> list:
    """List all files in the output directory."""
    files = []
    for f in sorted(OUTPUT_DIR.iterdir()):
        files.append({
            "name": f.name,
            "size": f.stat().st_size if f.is_file() else 0,
            "type": "file" if f.is_file() else "folder",
        })
    return files


def read_file(filename: str) -> str:
    """Read a file from output/ directory."""
    path = safe_path(filename)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
