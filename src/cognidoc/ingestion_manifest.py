"""
Ingestion manifest for tracking processed source files.

Enables incremental ingestion by identifying new/modified files
so only those files are re-processed through the pipeline.
"""

import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from .utils.logger import logger

MANIFEST_VERSION = 1


@dataclass
class FileRecord:
    """Record of an ingested source file."""

    path: str  # Relative path from sources_dir
    stem: str  # PDF stem (used for filtering downstream stages)
    size: int  # File size in bytes
    mtime: float  # Modification time
    content_hash: str  # SHA-256 of file content
    ingested_at: str  # ISO timestamp of ingestion


@dataclass
class IngestionManifest:
    """Tracks which source files have been ingested."""

    files: Dict[str, FileRecord] = field(default_factory=dict)
    created_at: str = ""
    last_updated: str = ""
    version: int = MANIFEST_VERSION

    @classmethod
    def load(cls, manifest_path: Path) -> Optional["IngestionManifest"]:
        """Load manifest from disk. Returns None if not found or invalid."""
        path = Path(manifest_path)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = cls(
                created_at=data.get("created_at", ""),
                last_updated=data.get("last_updated", ""),
                version=data.get("version", 1),
            )
            for rel_path, record_data in data.get("files", {}).items():
                manifest.files[rel_path] = FileRecord(**record_data)
            logger.info(f"Loaded ingestion manifest: {len(manifest.files)} files tracked")
            return manifest
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load ingestion manifest: {e}")
            return None

    def save(self, manifest_path: Path) -> None:
        """Save manifest to disk atomically (write to temp, then rename)."""
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        self.last_updated = now

        data = {
            "version": self.version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "files": {rel_path: asdict(record) for rel_path, record in self.files.items()},
        }

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix="manifest_")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(tmp_path, str(path))
            logger.info(f"Ingestion manifest saved: {len(self.files)} files tracked")
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def get_new_and_modified_files(
        self,
        sources_dir: Path,
        source_files: Optional[List[str]] = None,
    ) -> Tuple[List[Path], List[Path], Set[str]]:
        """
        Compare current source files against manifest.

        Args:
            sources_dir: Path to the sources directory
            source_files: Optional list of specific file paths to check.
                          If None, scans the entire sources_dir.

        Returns:
            (new_files, modified_files, all_new_stems)
        """
        sources_dir = Path(sources_dir)
        new_files: List[Path] = []
        modified_files: List[Path] = []
        all_stems: Set[str] = set()

        # Determine which files to check
        if source_files:
            candidates = [Path(f) for f in source_files if Path(f).is_file()]
        else:
            candidates = [f for f in sources_dir.rglob("*") if f.is_file()]

        for file_path in candidates:
            try:
                rel_path = str(file_path.relative_to(sources_dir))
            except ValueError:
                # File is outside sources_dir (absolute path passed directly)
                rel_path = file_path.name

            stem = file_path.stem

            if rel_path not in self.files:
                new_files.append(file_path)
                all_stems.add(stem)
                continue

            record = self.files[rel_path]
            stat = file_path.stat()

            # Fast path: check size and mtime first
            if stat.st_size != record.size or stat.st_mtime != record.mtime:
                # Confirm with content hash
                current_hash = self.compute_file_hash(file_path)
                if current_hash != record.content_hash:
                    modified_files.append(file_path)
                    all_stems.add(stem)

        return new_files, modified_files, all_stems

    def record_file(self, file_path: Path, sources_dir: Path, stem: str) -> None:
        """Record a file as successfully ingested."""
        file_path = Path(file_path)
        try:
            rel_path = str(file_path.relative_to(sources_dir))
        except ValueError:
            rel_path = file_path.name

        stat = file_path.stat()
        self.files[rel_path] = FileRecord(
            path=rel_path,
            stem=stem,
            size=stat.st_size,
            mtime=stat.st_mtime,
            content_hash=self.compute_file_hash(file_path),
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )

    def record_all_sources(self, sources_dir: Path) -> None:
        """Record all files in sources_dir (for full ingestion)."""
        sources_dir = Path(sources_dir)
        for file_path in sources_dir.rglob("*"):
            if file_path.is_file():
                self.record_file(file_path, sources_dir, file_path.stem)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
