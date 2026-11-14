"""Resolve ranked rows and read native RGB windows for object detection."""

from collections import defaultdict
from functools import lru_cache

from storage import contained


def identity(match):
    dataset, shard, row_id, _ = match
    return dataset.dataset_id, shard, row_id


@lru_cache(maxsize=16)
def _transformer(crs_wkt):
    from pyproj import Transformer

    return Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)


def box_geometry(row, box):
    """Map chip-relative pixel corners through the published window affine."""
    from affine import Affine

    left, top, right, bottom = box
    transform = Affine(*row["window_transform"])
    corners = [(left, top), (left, bottom), (right, bottom), (right, top)]
    if transform.determinant > 0:
        corners.reverse()
    corners.append(corners[0])
    project = _transformer(row["crs_wkt"]).transform
    return {
        "type": "Polygon",
        "coordinates": [[project(*(transform * corner), errcheck=True) for corner in corners]],
    }


class SearchChips:
    """Request-local metadata and checksum cache; no imagery is written."""

    def __init__(self):
        self.rows = {}
        self.verified_sources = {}

    def load_rows(self, matches):
        from geospatial_search.bundles import selected_rows

        missing = defaultdict(list)
        for match in matches:
            if identity(match) not in self.rows:
                missing[(match[0], match[1])].append(match[2])
        for (dataset, shard), ids in missing.items():
            root = dataset.directory / "shards"
            path = contained(root, root / shard / "chips.geoparquet")
            for row_id, row in selected_rows(path, ids).items():
                self.rows[(dataset.dataset_id, shard, row_id)] = row

    def source(self, dataset, row):
        from geospatial_search.download_nys_naip import sha256

        path = contained(dataset.imagery, dataset.imagery / row["source_uri"])
        if path not in self.verified_sources:
            self.verified_sources[path] = sha256(path)
        if self.verified_sources[path] != row["source_sha256"]:
            raise ValueError("Result source checksum differs")
        return path

    def images(self, matches):
        import rasterio
        from rasterio.windows import Window

        from geospatial_search.tiling import read_rgb

        self.load_rows(matches)
        by_source = defaultdict(list)
        for index, match in enumerate(matches):
            row = self.rows[identity(match)]
            if (row["pixel_width"], row["pixel_height"]) != (256, 256):
                raise ValueError("Detection requires a native 256 x 256 embedding window")
            source = self.source(match[0], row)
            by_source[source].append((index, Window(row["pixel_x"], row["pixel_y"], 256, 256)))
        images = [None] * len(matches)
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            for path, rows in by_source.items():
                with rasterio.open(path, "r") as dataset:
                    for index, window in rows:
                        images[index] = read_rgb(dataset, window)
        return images

    def results(self, matches, detections):
        from shapely import from_wkb

        self.load_rows(matches)
        results = []
        fields = (
            "embedding_id",
            "chip_id",
            "source_sha256",
            "pixel_x",
            "pixel_y",
            "pixel_width",
            "pixel_height",
            "centroid_lon",
            "centroid_lat",
            "west",
            "south",
            "east",
            "north",
            "crs_wkt",
            "scale_id",
            "window_transform",
        )
        for rank, match in enumerate(matches, 1):
            dataset, shard, _, score = match
            key = identity(match)
            row = self.rows[key]
            self.source(dataset, row)
            source_uri = row["source_uri"]
            result = {
                "rank": rank,
                "score": score,
                "shard_id": shard,
                **{name: row[name] for name in fields},
                "dataset_id": dataset.dataset_id,
                "dataset_source_uri": source_uri,
                "source_uri": dataset.qualify(source_uri),
                "release_year": dataset.release_year,
                "acquisition_date": dataset.acquisition_date(source_uri),
                "geometry": from_wkb(row["geometry"]).__geo_interface__,
            }
            if key in detections:
                detection = detections[key]
                result["detection"] = {
                    **detection,
                    "predictions": [
                        {**prediction, "geometry": box_geometry(row, prediction["box"])}
                        for prediction in detection["predictions"]
                    ],
                }
            results.append(result)
        return results
