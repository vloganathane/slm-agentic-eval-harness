"""Dataset downloader agents: Hugging Face, Kaggle, direct URLs."""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from slm_eval.errors import SlmEvalError


class DataDownloader(ABC):
    """Abstract base for dataset downloaders."""

    @abstractmethod
    def download(self, source: str, dest: Path) -> Path:
        """Download dataset from source to dest. Returns path to downloaded data."""
        pass


class HFDownloader(DataDownloader):
    """Hugging Face Hub dataset downloader."""

    def __init__(self, cache_dir: str | Path | None = None, token: str | None = None):
        if not HF_AVAILABLE:
            raise SlmEvalError("datasets library not installed. Run: pip install datasets")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/hf_datasets")
        self.token = token
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, source: str, dest: Path) -> Path:
        """
        Download HF dataset.

        Source format: "org/dataset_name" or "org/dataset_name@revision"
        or full HF Hub URL.
        """
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https") and "huggingface.co" in parsed.netloc:
            # Extract dataset name from URL
            path_parts = parsed.path.strip("/").split("/")
            if "datasets" in path_parts:
                idx = path_parts.index("datasets")
                if idx + 1 < len(path_parts):
                    dataset_id = "/".join(path_parts[idx+1:idx+3])
                else:
                    raise SlmEvalError(f"Cannot parse HF dataset URL: {source}")
            else:
                raise SlmEvalError(f"Cannot parse HF dataset URL: {source}")
        else:
            dataset_id = source

        dest.mkdir(parents=True, exist_ok=True)

        # Check if revision specified
        revision = "main"
        if "@" in dataset_id:
            dataset_id, revision = dataset_id.split("@", 1)

        print(f"[HFDownloader] Loading {dataset_id}@{revision} from HF Hub...")

        try:
            ds = load_dataset(
                dataset_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                token=self.token,
            )
        except Exception as e:
            raise SlmEvalError(f"Failed to load HF dataset {dataset_id}: {e}") from e

        # Save each split to dest
        for split_name, split_ds in ds.items():
            out_path = dest / f"{split_name}.parquet"
            print(f"[HFDownloader] Saving split '{split_name}' ({len(split_ds)} rows) to {out_path}")
            split_ds.to_parquet(out_path)

        return dest


class KaggleDownloader(DataDownloader):
    """Kaggle dataset downloader via CLI."""

    def __init__(self, api_key: str | None = None, data_dir: str | Path | None = None):
        self.api_key = api_key or os.environ.get("KAGGLE_API_KEY")
        self.data_dir = Path(data_dir) if data_dir else Path("/tmp/kaggle_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.api_key:
            self._configure_kaggle()

    def _configure_kaggle(self):
        """Write kaggle.json to ~/.kaggle/"""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        if not kaggle_json.exists():
            import json
            # Expect API key format: "username:key"
            if ":" in self.api_key:
                username, key = self.api_key.split(":", 1)
            else:
                # Fallback: assume it's just the key, try to get username from env
                username = os.environ.get("KAGGLE_USERNAME", "unknown")
                key = self.api_key
            kaggle_json.write_text(json.dumps({"username": username, "key": key}))
            kaggle_json.chmod(0o600)

    def download(self, source: str, dest: Path) -> Path:
        """
        Download Kaggle dataset.

        Source format: "username/dataset-name" or "username/dataset-name/version"
        or full Kaggle URL.
        """
        if not shutil.which("kaggle"):
            raise SlmEvalError("kaggle CLI not installed. Run: pip install kaggle")

        parsed = urlparse(source)
        if parsed.scheme in ("http", "https") and "kaggle.com" in parsed.netloc:
            path_parts = parsed.path.strip("/").split("/")
            if "datasets" in path_parts:
                idx = path_parts.index("datasets")
                if idx + 2 < len(path_parts):
                    dataset_ref = f"{path_parts[idx+1]}/{path_parts[idx+2]}"
                else:
                    raise SlmEvalError(f"Cannot parse Kaggle dataset URL: {source}")
            else:
                raise SlmEvalError(f"Cannot parse Kaggle dataset URL: {source}")
        else:
            dataset_ref = source

        dest.mkdir(parents=True, exist_ok=True)

        print(f"[KaggleDownloader] Downloading {dataset_ref}...")

        try:
            cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(dest), "--unzip"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise SlmEvalError(f"Kaggle download failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise SlmEvalError("Kaggle download timed out (10 min)")
        except Exception as e:
            raise SlmEvalError(f"Kaggle download error: {e}") from e

        print(f"[KaggleDownloader] Downloaded to {dest}")
        return dest


class URLDownloader(DataDownloader):
    """Fallback downloader for direct HTTP/HTTPS URLs."""

    def download(self, source: str, dest: Path) -> Path:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(source)
        filename = Path(parsed.path).name or "dataset"
        out_path = dest / filename

        print(f"[URLDownloader] Downloading {source} -> {out_path}")

        try:
            urllib.request.urlretrieve(source, out_path)
        except Exception as e:
            raise SlmEvalError(f"URL download failed: {e}") from e

        # Auto-extract if archive
        if out_path.suffix in (".zip", ".tar", ".gz", ".tgz"):
            self._extract(out_path, dest)
            out_path.unlink()  # remove archive after extraction

        # If file has no extension, try to detect if it's JSON and rename
        if not out_path.suffix:
            # Check if it's valid JSON
            try:
                import json
                with out_path.open("r", encoding="utf-8") as f:
                    json.load(f)
                # It's JSON - rename with .json extension
                new_path = out_path.with_suffix(".json")
                out_path.rename(new_path)
                out_path = new_path
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not JSON, leave as-is
                pass

        return dest

    def _extract(self, archive: Path, dest: Path):
        import tarfile
        import zipfile

        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest)
        elif archive.suffix in (".tar", ".gz", ".tgz"):
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest)
        else:
            raise SlmEvalError(f"Unsupported archive format: {archive.suffix}")


class AutoDownloader:
    """Automatically selects the right downloader based on source URL/format."""

    def __init__(
        self,
        hf_cache_dir: str | Path | None = None,
        hf_use_auth_token: bool = False,
        kaggle_api_key: str | None = None,
        kaggle_data_dir: str | Path | None = None,
    ):
        self.hf = HFDownloader(hf_cache_dir, hf_use_auth_token) if HF_AVAILABLE else None
        self.kaggle = KaggleDownloader(kaggle_api_key, kaggle_data_dir)
        self.url = URLDownloader()

    def download(self, source: str, dest: Path) -> Path:
        """Auto-detect source type and download."""
        parsed = urlparse(source)

        if parsed.scheme in ("http", "https"):
            if "huggingface.co" in parsed.netloc and self.hf:
                return self.hf.download(source, dest)
            elif "kaggle.com" in parsed.netloc:
                return self.kaggle.download(source, dest)
            else:
                return self.url.download(source, dest)
        elif ("/" in source or source.count("/") == 0) and not source.startswith("."):
            # Heuristic: "org/dataset" or "dataset" -> try HF first, then Kaggle
            if self.hf:
                try:
                    return self.hf.download(source, dest)
                except SlmEvalError:
                    pass
            try:
                return self.kaggle.download(source, dest)
            except SlmEvalError:
                pass
            raise SlmEvalError(f"Could not download {source} from HF or Kaggle")
        else:
            raise SlmEvalError(f"Unrecognized source format: {source}")
