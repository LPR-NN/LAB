"""Audit storage for decision packages."""

from pathlib import Path
from datetime import datetime

from source.contracts.audit import DecisionPackage
from source.audit.package import compute_package_hash


class AuditStorage:
    """
    Storage for decision audit packages.

    Saves packages as JSON files with integrity hashes.
    """

    def __init__(self, storage_path: Path | str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, package: DecisionPackage) -> Path:
        """
        Save a decision package.

        Args:
            package: The package to save

        Returns:
            Path to the saved file
        """
        # Compute hash
        content_hash = compute_package_hash(package)
        package.content_hash = content_hash

        # Serialize with deterministic formatting
        json_content = package.model_dump_json(indent=2)

        # Save package
        file_path = self.storage_path / f"{package.package_id}.json"
        file_path.write_text(json_content, encoding="utf-8")

        # Save hash file
        hash_path = self.storage_path / f"{package.package_id}.sha256"
        hash_path.write_text(content_hash, encoding="utf-8")

        return file_path

    def load(self, package_id: str) -> DecisionPackage | None:
        """
        Load a decision package by ID.

        Args:
            package_id: The package UUID

        Returns:
            Loaded package or None if not found
        """
        file_path = self.storage_path / f"{package_id}.json"

        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")
        return DecisionPackage.model_validate_json(content)

    def verify(self, package_id: str) -> tuple[bool, str]:
        """
        Verify integrity of a stored package.

        Args:
            package_id: The package UUID

        Returns:
            (is_valid, message) tuple
        """
        package = self.load(package_id)
        if package is None:
            return False, f"Package {package_id} not found"

        # Load stored hash
        hash_path = self.storage_path / f"{package_id}.sha256"
        if not hash_path.exists():
            return False, "Hash file missing"

        stored_hash = hash_path.read_text(encoding="utf-8").strip()

        # Compute current hash
        current_hash = compute_package_hash(package)

        if stored_hash == current_hash:
            return True, "Package integrity verified"
        else:
            return False, "Hash mismatch - package may have been modified"

    def list_packages(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[str]:
        """
        List all package IDs, optionally filtered by date.

        Args:
            start_date: Filter packages after this date
            end_date: Filter packages before this date

        Returns:
            List of package IDs
        """
        packages: list[str] = []

        for file_path in self.storage_path.glob("*.json"):
            package_id = file_path.stem

            # Skip hash files
            if file_path.suffix == ".sha256":
                continue

            if start_date or end_date:
                # Load package to check timestamp
                package = self.load(package_id)
                if package:
                    if start_date and package.timestamp < start_date:
                        continue
                    if end_date and package.timestamp > end_date:
                        continue

            packages.append(package_id)

        return sorted(packages)

    def delete(self, package_id: str) -> bool:
        """
        Delete a package.

        Args:
            package_id: The package UUID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_path / f"{package_id}.json"
        hash_path = self.storage_path / f"{package_id}.sha256"

        deleted = False

        if file_path.exists():
            file_path.unlink()
            deleted = True

        if hash_path.exists():
            hash_path.unlink()

        return deleted

    def get_statistics(self) -> dict:
        """Get storage statistics."""
        packages = self.list_packages()

        total_size = sum(
            (self.storage_path / f"{p}.json").stat().st_size
            for p in packages
            if (self.storage_path / f"{p}.json").exists()
        )

        return {
            "total_packages": len(packages),
            "total_size_bytes": total_size,
            "storage_path": str(self.storage_path),
        }


__all__ = ["AuditStorage"]
