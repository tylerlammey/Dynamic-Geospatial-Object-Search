"""Record the assistant's pre-inference visual audit, never human ground truth.

These small, fallible convenience labels support exploratory comparisons only.
They cannot establish detector accuracy, calibration, or a 99% recall guarantee.
Re-run before inference; original native review sheets are retained locally.
"""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Eight independently inspected tiles per query. None means visually ambiguous.
LABELS = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # fighter jet: large transports are not fighters
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],  # runway differs from taxiway/apron
    [1, 1, 1, 1, 1, None, None, 0],
    [1, 1, 1, 1, 1, 1, 1, None],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
]
# Visible-water pool bounds, axis aligned/clipped to tile; approximate hand audit.
# A partial boundary object counts as present. Pools are separate from buildings,
# decks, shadows and wakes; circular baby pool is a separate swimming pool.
POOL_BOXES = {
    64: [[0, 27, 75, 220]],
    65: [[111, 126, 183, 212]],
    66: [[88, 132, 162, 245], [176, 181, 196, 201]],
    67: [[114, 0, 202, 46], [77, 95, 137, 143]],
    68: [[112, 56, 140, 110], [157, 0, 252, 95]],
    69: [[83, 176, 149, 251]],
    70: [[125, 11, 170, 56]],
    71: [],
}
AIRPLANE_BOXES = {
    0: [[70, 117, 194, 228]],
    1: [[0, 91, 69, 159]],
    2: [[24, 73, 95, 185], [144, 69, 226, 171]],
    3: [[0, 70, 36, 169]],
}


def main():
    target = ROOT / "local/tiles.json"
    manifest = json.loads(target.read_text())
    # Idempotent; extra query pairs reference the same pixels, never duplicate TIFFs.
    rows = manifest["rows"][:96]
    for i, row in enumerate(rows):
        row.update(
            label=LABELS[i // 8][i % 8],
            annotation_source="assistant_visual_audit_before_candidate_inference",
            annotation_quality="exploratory_unverified",
            boxes=[],
            exhaustive=False,
        )
        if i in POOL_BOXES:
            row.update(boxes=POOL_BOXES[i], exhaustive=True)
    extras = []
    for index in range(8):
        row = copy.deepcopy(rows[index])
        row.update(
            query="airplane",
            label=int(index < 4),
            boxes=AIRPLANE_BOXES.get(index, []),
            exhaustive=True,
            stratum="paired_query_audit",
        )
        row["sample_id"] = hashlib.sha256((row["tile_id"] + "|airplane").encode()).hexdigest()[:20]
        extras.append(row)
    for index, box in [(4, [84, 129, 104, 159]), (5, [218, 52, 245, 65]), (6, None), (7, None)]:
        row = copy.deepcopy(rows[index])
        row.update(
            query="boat",
            label=int(box is not None),
            boxes=[box] if box else [],
            exhaustive=True,
            stratum="paired_query_small_object_audit",
        )
        row["sample_id"] = hashlib.sha256((row["tile_id"] + "|boat").encode()).hexdigest()[:20]
        extras.append(row)
    manifest["rows"] = rows + extras
    manifest["annotation"] = {
        "source": "assistant visual audit",
        "date": "2026-10-26",
        "human_verified": False,
        "performed_before_inference": True,
        "limitations": "Convenience sample; approximate boxes; ambiguous labels excluded; not evidence of ultra-high recall.",
        "bbox_convention": "xyxy native pixels, visible object bounds clipped to tile; no rotated/crowd labels",
    }
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    # Red/green image overlays remain local for human review, never a new dataset.
    from dataset import load_images
    from PIL import Image, ImageDraw

    selected = [r for r in manifest["rows"] if r["exhaustive"]]
    images, _ = load_images({"rows": selected})
    canvas = Image.new("RGB", (5 * 256, 4 * 280), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (row, image) in enumerate(zip(selected, images)):
        x, y = i % 5 * 256, i // 5 * 280
        canvas.paste(image, (x, y))
        for box in row["boxes"]:
            draw.rectangle([x + box[0], y + box[1], x + box[2], y + box[3]], outline="red", width=1)
        draw.text((x + 2, y + 258), f"{i}: {row['query']} | assistant audit", fill="black")
    canvas.save(ROOT / "local/annotation-audit.jpg", quality=97)
    print(len(manifest["rows"]), "pairs;", len(selected), "exhaustive exploratory box annotations")


if __name__ == "__main__":
    main()
