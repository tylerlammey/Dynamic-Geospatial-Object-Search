/** Wire contracts for the local service. Coordinates are longitude first. */
import type {Polygon} from 'geojson';

export type Bounds = [number, number, number, number];

export type DetectionOptions = {
  enabled: boolean;
  box_threshold: number;
  max_candidates: number;
};

export type DetectionBox = {
  box: [number, number, number, number];
  score: number;
  text_label: string;
  geometry: Polygon;
  touches_tile_edge?: boolean;
};

export type SearchProgress = {
  state: string;
  error?: string;
  result_id?: string;
  scanned_sources?: number;
  searched_source_tiles?: number;
  evaluated_candidates?: number;
  accepted_candidates?: number;
  max_candidates?: number;
  retrieval_round?: number;
  recent_detection_percent?: number | null;
};

export type Chip = {
  rank: number | null;
  score: number | null;
  shard_id: string;
  embedding_id: number | null;
  chip_id: string;
  source_uri: string;
  source_sha256: string;
  dataset_id?: string;
  release_year?: number | null;
  acquisition_date?: string | null;
  pixel_x: number;
  pixel_y: number;
  pixel_width: number;
  pixel_height: number;
  centroid_lon: number;
  centroid_lat: number;
  west: number;
  south: number;
  east: number;
  north: number;
  geometry: Polygon;
  crs_wkt: string;
  detection?: {predictions: DetectionBox[]; maximum_box_score: number; score_kind: string};
};

export type ResultSet = {
  query: string;
  result_id?: string;
  run_fingerprint: string;
  results: Chip[];
  searched_source_tiles: number;
  total_source_tiles: number;
  searched_chip_count: number;
  searched_source_uris?: string[];
  partial_collection: boolean;
  created_utc: number;
  search?: {backend: string; approximate: boolean; search_seconds: number};
  detection?: {
    enabled: boolean;
    model_id?: string;
    box_threshold: number;
    text_threshold: number;
    evaluated: number;
    accepted: number;
    rounds: number;
    returned: number;
    max_candidates: number;
    recent_detection_percent?: number | null;
    stop_reason: string;
  };
};

export type Status = {
  state: string;
  collection_state: string;
  busy: boolean;
  published_sources: number;
  published_chips: number;
  total_sources: number;
  coverage: string[];
  run_fingerprint?: string;
  partial: boolean;
  error?: string;
  imagery_error?: string;
  model?: {state: string; error?: string; gpu?: string};
  cache: {bytes: number; limit_bytes: number; files: number};
  elapsed_hms?: string;
  eta_hms?: string;
  search_index?: {ready: boolean; algorithm: string};
  overview?: {
    state: string;
    ready: boolean;
    epoch: number;
    completed_sources?: number;
    total_sources?: number;
    eta_seconds?: number;
    error?: string;
  };
};

export type Configuration = {
  synthetic?: boolean;
  detection_defaults?: Pick<DetectionOptions, 'box_threshold' | 'max_candidates'> & {
    min_candidates: number;
    min_detection_rate: number;
  };
  imagery: {
    bounds: Bounds;
    count: number;
    datasets?: {dataset_id: string; release_year: number | null; sources: number}[];
  } | null;
  imagery_error?: string;
  sources: {uri: string; bounds: Bounds}[];
  attribution: string;
};
