"""
Data downloading and processing utilities for NASA EPIC satellite images.

API reference: https://epic.gsfc.nasa.gov/about/api

Endpoints:
  Metadata:  https://epic.gsfc.nasa.gov/api/natural/available
             https://epic.gsfc.nasa.gov/api/natural/date/YYYY-MM-DD
  Images:    https://epic.gsfc.nasa.gov/archive/natural/YYYY/MM/DD/png/{name}.png
"""

import asyncio
import json
import ssl
import aiohttp
import certifi
import requests
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

API_BASE = "https://epic.gsfc.nasa.gov/api/natural"
ARCHIVE_BASE = "https://epic.gsfc.nasa.gov/archive/natural"

MAX_CONCURRENT_DOWNLOADS = 10


def _create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context using certifi's CA bundle (fixes macOS issues)."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    return ctx


class EPICDataDownloader:
    """Downloads metadata and images from the NASA EPIC API."""

    def __init__(self, config):
        self.config = config.data
        self.images_dir = Path(self.config.images_dir)
        self.combined_dir = Path(self.config.combined_dir)

    # ── Sync helpers (single requests, used for simple one-off calls) ──

    def fetch_available_dates(self) -> List[str]:
        """Fetch list of all available dates from the API."""
        url = f"{API_BASE}/available"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dates = response.json()
        dates.sort()
        logger.info(f"Found {len(dates)} available dates")
        return dates

    def fetch_date_metadata(self, date: str) -> List[dict]:
        """Fetch full metadata for a specific date and save to combined dir."""
        url = f"{API_BASE}/date/{date}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        self.combined_dir.mkdir(parents=True, exist_ok=True)
        with open(self.combined_dir / f"{date}.json", "w") as f:
            json.dump(data, f)

        logger.info(f"Fetched metadata for {date}: {len(data)} images")
        return data

    # ── Async internals ──

    async def _async_download_image(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
        image_name: str, date: str,
    ) -> bool:
        """Download a single image, respecting the concurrency semaphore."""
        if not image_name.endswith(".png"):
            image_name = f"{image_name}.png"

        date_dir = self.images_dir / date
        image_path = date_dir / image_name

        if image_path.exists():
            return True

        year, month, day = date[:4], date[5:7], date[8:10]
        url = f"{ARCHIVE_BASE}/{year}/{month}/{day}/png/{image_name}"

        try:
            async with semaphore:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content = await resp.read()

            date_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = image_path.with_suffix('.tmp')
            with open(tmp_path, "wb") as f:
                f.write(content)
            tmp_path.rename(image_path)

            logger.debug(f"Downloaded: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {image_name}: {e}")
            return False

    async def _async_download_date_images(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
        date: str,
    ) -> Tuple[int, int]:
        """Download all images for a date concurrently."""
        metadata_path = self.combined_dir / f"{date}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
        else:
            # Fetch metadata through the async session
            async with semaphore:
                async with session.get(f"{API_BASE}/date/{date}") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            self.combined_dir.mkdir(parents=True, exist_ok=True)
            with open(self.combined_dir / f"{date}.json", "w") as f:
                json.dump(data, f)

        images = [item["image"] for item in data if item.get("image")]
        tasks = [
            self._async_download_image(session, semaphore, name, date)
            for name in images
        ]
        results = await asyncio.gather(*tasks)
        success = sum(1 for r in results if r)
        logger.info(f"{date}: downloaded {success}/{len(images)} images")
        return success, len(images)

    async def _async_download_recent(self, num_days: int) -> bool:
        """Download images from the most recent N days concurrently."""
        dates = self.fetch_available_dates()
        recent_dates = dates[-num_days:]
        logger.info(f"Downloading images for {len(recent_dates)} most recent dates")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ssl=_create_ssl_context())

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                self._async_download_date_images(session, semaphore, date)
                for date in recent_dates
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        total_success, total_count = 0, 0
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Date download failed: {r}")
            else:
                total_success += r[0]
                total_count += r[1]

        logger.info(f"Recent download complete: {total_success}/{total_count} images")
        return total_success > 0

    async def _async_download_metadata(self) -> bool:
        """Download metadata for all available dates concurrently."""
        dates = self.fetch_available_dates()
        logger.info(f"Downloading metadata for {len(dates)} dates")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ssl=_create_ssl_context())

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async def _fetch_one(date: str):
                metadata_path = self.combined_dir / f"{date}.json"
                if metadata_path.exists():
                    return
                async with semaphore:
                    async with session.get(f"{API_BASE}/date/{date}") as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                self.combined_dir.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(data, f)
                logger.info(f"Fetched metadata for {date}: {len(data)} images")

            tasks = [_fetch_one(date) for date in dates]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            for f in failures:
                logger.error(f"Metadata download failed: {f}")
            return False

        logger.info("Metadata download complete")
        return True

    # ── Public sync API (callers don't need to change) ──

    def download_image(self, image_name: str, date: str) -> bool:
        """Download a single image (sync, for one-off use)."""
        if not image_name.endswith(".png"):
            image_name = f"{image_name}.png"

        date_dir = self.images_dir / date
        image_path = date_dir / image_name

        if image_path.exists():
            return True

        year, month, day = date[:4], date[5:7], date[8:10]
        url = f"{ARCHIVE_BASE}/{year}/{month}/{day}/png/{image_name}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download {image_name}: {e}")
            return False

        date_dir.mkdir(parents=True, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(response.content)

        logger.debug(f"Downloaded: {image_path}")
        return True

    def download_date_images(self, date: str) -> Tuple[int, int]:
        """Download all images for a date concurrently."""
        return asyncio.run(self._async_download_date_single(date))

    async def _async_download_date_single(self, date: str) -> Tuple[int, int]:
        """Helper: open a session and download one date's images."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ssl=_create_ssl_context())
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            return await self._async_download_date_images(session, semaphore, date)

    def download_metadata(self) -> bool:
        """Download metadata for all available dates concurrently."""
        return asyncio.run(self._async_download_metadata())

    def download_recent(self, num_days: int = 7) -> bool:
        """Download images from the most recent N days concurrently."""
        return asyncio.run(self._async_download_recent(num_days))

    # ── Utilities ──

    def scan_available_data(self) -> Dict[str, Any]:
        """Scan local images directory and return statistics."""
        stats = {
            "total_dates": 0,
            "total_images": 0,
            "available_dates": [],
            "images_per_date": {},
        }

        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return stats

        for date_dir in sorted(self.images_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name == ".DS_Store":
                continue
            image_files = list(date_dir.glob("*.png"))
            if image_files:
                stats["available_dates"].append(date_dir.name)
                stats["images_per_date"][date_dir.name] = len(image_files)
                stats["total_images"] += len(image_files)

        stats["total_dates"] = len(stats["available_dates"])
        logger.info(f"Found {stats['total_images']} images across {stats['total_dates']} dates")
        return stats


class CoordinateExtractor:
    """Extracts coordinate data from saved metadata in combined/ directory."""

    def __init__(self, config):
        self.combined_dir = Path(config.data.combined_dir)

    def extract_coordinates(self) -> Tuple[List[float], List[float]]:
        """Extract latitude and longitude from all metadata files."""
        lat_coordinates = []
        lon_coordinates = []

        if not self.combined_dir.exists():
            raise FileNotFoundError(f"Combined directory not found: {self.combined_dir}")

        for json_file in sorted(self.combined_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                for item in data:
                    coords = item.get("centroid_coordinates", {})
                    lat = coords.get("lat")
                    lon = coords.get("lon")
                    if lat is not None and lon is not None:
                        lat_coordinates.append(float(lat))
                        lon_coordinates.append(float(lon))

            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")

        logger.info(f"Extracted {len(lat_coordinates)} coordinate pairs")
        return lat_coordinates, lon_coordinates

    def get_coordinate_stats(self, lat_coords: List[float], lon_coords: List[float]) -> pd.DataFrame:
        """Get statistical summary of coordinates."""
        stats = pd.DataFrame({
            "latitude": pd.Series(lat_coords).describe(),
            "longitude": pd.Series(lon_coords).describe(),
        })
        return stats
