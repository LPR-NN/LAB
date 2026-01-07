"Corpus deduplication - find and remove duplicate files."

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from source.corpus.constants import CORPUS_EXTENSIONS


@dataclass
class DuplicateGroup:
    content_hash: str
    files: list[Path]
    size_bytes: int

    @property
    def keeper(self) -> Path:
        return min(self.files, key=lambda p: (len(str(p)), str(p)))

    @property
    def duplicates(self) -> list[Path]:
        keeper = self.keeper
        return [f for f in self.files if f != keeper]


class CorpusDeduplicator:
    """
    Find and remove duplicate files in corpus by content hash.

    Keeps the file with shortest path (tiebreaker: alphabetically first).
    """

    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path

    def _compute_hash(self, file_path: Path) -> str:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _find_corpus_files(self) -> list[Path]:
        files: list[Path] = []
        for ext in CORPUS_EXTENSIONS:
            files.extend(self.corpus_path.rglob(ext))
        return sorted(files)

    def find_duplicates(self) -> list[DuplicateGroup]:
        """
        Find all duplicate groups in corpus.

        Returns list of DuplicateGroup, each containing 2+ files with identical content.
        """
        hash_to_files: dict[str, list[Path]] = defaultdict(list)

        for file_path in self._find_corpus_files():
            try:
                content_hash = self._compute_hash(file_path)
                hash_to_files[content_hash].append(file_path)
            except Exception:
                continue

        groups: list[DuplicateGroup] = []
        for content_hash, files in hash_to_files.items():
            if len(files) > 1:
                size = files[0].stat().st_size
                groups.append(
                    DuplicateGroup(
                        content_hash=content_hash,
                        files=sorted(files),
                        size_bytes=size,
                    )
                )

        return sorted(groups, key=lambda g: -len(g.files))

    def remove_duplicates(self, dry_run: bool = True) -> tuple[int, int]:
        print(f"Removing duplicates from {self.corpus_path} (dry_run: {dry_run})")
        """
        Remove duplicate files, keeping one per group.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            (removed_count, bytes_freed)
        """
        groups = self.find_duplicates()
        removed = 0
        bytes_freed = 0

        for group in groups:
            for dup in group.duplicates:
                if dry_run:
                    print(f"  Would remove: {dup}")
                else:
                    try:
                        dup.unlink()
                        print(f"  Removed: {dup}")
                    except Exception as e:
                        print(f"  Failed to remove {dup}: {e}")
                        continue
                removed += 1
                bytes_freed += group.size_bytes

        return removed, bytes_freed

    def get_statistics(self) -> dict[str, int]:
        """Get duplicate statistics without removing anything."""
        groups = self.find_duplicates()
        total_duplicates = sum(len(g.duplicates) for g in groups)
        total_bytes = sum(g.size_bytes * len(g.duplicates) for g in groups)

        return {
            "duplicate_groups": len(groups),
            "total_duplicates": total_duplicates,
            "bytes_wasted": total_bytes,
        }


__all__ = ["CorpusDeduplicator", "DuplicateGroup"]
