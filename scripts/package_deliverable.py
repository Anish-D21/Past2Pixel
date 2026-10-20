"""
Packages the Phase 1 deliverable ZIP archive excluding caches, datasets, and large checkpoints.
"""
import zipfile
from pathlib import Path

def create_phase_archive(phase_name: str = "phase1"):
    root = Path(__file__).resolve().parent.parent
    out_zip = root / f"{phase_name}_deliverable.zip"
    included = [
        "src",
        "scripts",
        "tests",
        "docs",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "run_gpu_training.bat",
        "ReadMe.md",
    ]

    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in included:
            p = root / item
            if p.is_file():
                zf.write(p, p.relative_to(root))
                count += 1
            elif p.is_dir():
                for f in p.rglob("*"):
                    if "__pycache__" not in f.parts and not f.name.endswith(".pyc") and f.is_file():
                        zf.write(f, f.relative_to(root))
                        count += 1

    size_kb = round(out_zip.stat().st_size / 1024, 2)
    print(f"Created archive: {out_zip}")
    print(f"Files included: {count}")
    print(f"Archive size:   {size_kb} KB")
    return out_zip

if __name__ == "__main__":
    create_phase_archive("phase1")
