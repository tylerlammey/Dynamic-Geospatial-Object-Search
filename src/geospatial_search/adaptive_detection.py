"""Continue retrieving ANN candidates while recent tiles yield detections."""

from collections import deque
from dataclasses import asdict, dataclass
from time import perf_counter

MAX_FETCH_SIZE = 512
MAX_CANDIDATES = 5000
CHIP_SIZE = 256


def _between(value, lower, upper):
    # Comparisons also reject NaN and infinity. Booleans are not scores.
    return type(value) in (int, float) and lower <= value <= upper


@dataclass(frozen=True)
class DetectionOptions:
    enabled: bool = True
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    nms_iou: float = 0.50
    batch_size: int = 2
    fetch_size: int = 32
    window_size: int = 32
    min_candidates: int = 64
    min_detection_rate: float = 0.05
    max_candidates: int = 256

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("detection.enabled must be boolean")
        for key in ("box_threshold", "text_threshold", "nms_iou", "min_detection_rate"):
            if not _between(getattr(self, key), 0, 1):
                raise ValueError(f"detection.{key} must be a finite number between 0 and 1")
        for key, upper in (
            ("batch_size", 2),
            ("fetch_size", MAX_FETCH_SIZE),
            ("window_size", MAX_CANDIDATES),
            ("min_candidates", MAX_CANDIDATES),
            ("max_candidates", MAX_CANDIDATES),
        ):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"detection.{key} must be an integer between 1 and {upper}")


def _validate_target(top_k):
    if type(top_k) is not int or not 1 <= top_k <= MAX_CANDIDATES:
        raise ValueError("Result count must be between 1 and 5,000")


def validate_options(value=None, top_k=25):
    """Validate detection settings and the legacy similarity-only result count."""
    _validate_target(top_k)
    if value is None:
        return DetectionOptions()
    if not isinstance(value, dict):
        raise ValueError("detection must be an object")
    unknown = value.keys() - DetectionOptions.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown detection option: {', '.join(sorted(map(str, unknown)))}")
    return DetectionOptions(**value)


def match_identity(match):
    dataset, shard, row, _ = match
    return dataset.dataset_id, shard, row


def _validate_detections(batch, count):
    if not isinstance(batch, (list, tuple)) or len(batch) != count:
        raise ValueError("Detector must return one result per candidate")
    for detection in batch:
        if (
            not isinstance(detection, dict)
            or not isinstance(detection.get("predictions"), list)
            or not _between(detection.get("maximum_box_score"), 0, 1)
            or not isinstance(detection.get("score_kind"), str)
            or not detection["score_kind"]
        ):
            raise ValueError("Detector returned invalid tile predictions")
        highest_score = 0.0
        for prediction in detection["predictions"]:
            if not isinstance(prediction, dict):
                raise ValueError("Detector returned an invalid prediction")
            box = prediction.get("box")
            if (
                not isinstance(box, (list, tuple))
                or len(box) != 4
                or not all(_between(value, 0, CHIP_SIZE) for value in box)
                or not box[0] < box[2]
                or not box[1] < box[3]
                or not _between(prediction.get("score"), 0, 1)
                or not isinstance(prediction.get("text_label"), str)
            ):
                raise ValueError("Detector returned an invalid chip box, score, or label")
            highest_score = max(highest_score, prediction["score"])
        if detection["maximum_box_score"] != highest_score:
            raise ValueError("Detector maximum box score differs from retained predictions")


def adaptive_search(fetch, evaluate, *, top_k, options, exclude_ids=(), progress=None):
    """Return every positive tile found before the recent detection rate drops.

    `fetch(count, exclusions)` returns federated matches plus retrieval diagnostics.
    Exclusions are qualified (dataset, shard, row) triples throughout the search.
    `evaluate(matches)` returns one validated prediction dictionary per match.
    `top_k` caps only similarity-only results when detection is disabled.
    Detection results are ordered by best box score, then original similarity.
    Return (matches, detections keyed by qualified ID, retrieval details, diagnostics).
    """
    _validate_target(top_k)
    if not options.enabled:
        matches, details = fetch(top_k, exclude_ids)
        return matches, {}, details, {"enabled": False}

    seen = {tuple(identifier) for identifier in exclude_ids}
    if any(len(identifier) != 3 for identifier in seen):
        raise ValueError("Adaptive retrieval exclusions must use qualified chip IDs")
    evaluated = 0
    box_count = 0
    accepted = []
    detections = {}
    recent = deque(maxlen=options.window_size)
    recent_positive = 0
    details = {}
    search_seconds = 0.0
    detection_seconds = 0.0
    approximate = False
    rounds = 0

    def detection_rate():
        return recent_positive / len(recent) if recent else None

    def update_progress():
        if progress is not None:
            rate = detection_rate()
            progress(
                {
                    "state": "detecting",
                    "evaluated_candidates": evaluated,
                    "accepted_candidates": len(accepted),
                    "max_candidates": options.max_candidates,
                    "box_count": box_count,
                    "retrieval_round": rounds,
                    "recent_detection_rate": rate,
                    "recent_detection_percent": None if rate is None else 100 * rate,
                }
            )

    stop_reason = None
    while stop_reason is None:
        count = min(options.fetch_size, options.max_candidates - evaluated)
        found, details = fetch(count, tuple(sorted(seen)))
        if len(found) > count:
            raise ValueError("Retrieval returned more than the requested candidates")
        rounds += 1
        approximate |= details.get("approximate", False)
        search_seconds += details.get("search_seconds", 0.0)

        fresh = []
        paged = set()
        for match in found:
            key = match_identity(match)
            if key not in seen and key not in paged:
                paged.add(key)
                fresh.append(match)
        if not fresh:
            stop_reason = "no_new_candidates"
            break

        for offset in range(0, len(fresh), options.batch_size):
            batch = fresh[offset : offset + options.batch_size]
            update_progress()
            started = perf_counter()
            output = evaluate(batch)
            detection_seconds += perf_counter() - started
            _validate_detections(output, len(batch))
            evaluated += len(batch)
            for match, detection in zip(batch, output, strict=True):
                key = match_identity(match)
                seen.add(key)
                positive = bool(detection["predictions"])
                if len(recent) == recent.maxlen:
                    recent_positive -= recent[0]
                recent.append(positive)
                recent_positive += positive
                if positive:
                    accepted.append(match)
                    detections[key] = detection
                    box_count += len(detection["predictions"])
            update_progress()
            if evaluated == options.max_candidates:
                stop_reason = "candidate_limit"
                break
            if (
                evaluated >= options.min_candidates
                and len(recent) == recent.maxlen
                and detection_rate() < options.min_detection_rate
            ):
                stop_reason = "low_detection_yield"
                break

    # Rank after retrieval so score order cannot change the detection rate window.
    # Python's stable sort keeps retrieval order for equal confidence and similarity.
    returned = sorted(
        accepted,
        key=lambda match: (-detections[match_identity(match)]["maximum_box_score"], -match[3]),
    )
    rate = detection_rate()
    diagnostics = {
        **asdict(options),
        "evaluated": evaluated,
        "accepted": len(accepted),
        "returned": len(returned),
        "box_count": box_count,
        "rounds": rounds,
        "recent_detection_rate": rate,
        "recent_detection_percent": None if rate is None else 100 * rate,
        "recent_evaluated": len(recent),
        "detection_seconds": detection_seconds,
        "stop_reason": stop_reason,
    }
    return (
        returned,
        detections,
        {
            **details,
            "search_seconds": search_seconds,
            "approximate": approximate,
            "retrieval_rounds": rounds,
        },
        diagnostics,
    )
