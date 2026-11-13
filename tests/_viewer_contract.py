"""An independent checker for the embeddings-tumbler data contract, schema version 1.

Written for this repo's tests from the contract document itself
(``docs/data-contract.md`` in agentculture/embeddings-tumbler, "Data contract,
schema version 1", frozen). It is **not** a copy of that repo's reference
validator and imports nothing from it: the point is a second reading of the
same document, so a misreading on either side shows up as a disagreement.

Scope: the file form, as a producer writes it -- a ``replace`` set (or an
``append`` checked on its own, without a base view). It implements checks 1 to
10 of the contract's check order; check 11 (append against a held view) needs a
viewer's state and does not apply to a producer's single file.

:func:`check_pair` returns the list of ``(code, detail)`` faults, empty when the
pair is accepted. ``detail`` is the offending field path for ``invalid-field``
and the relevant value otherwise.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

KIND = "embeddings-tumbler/data"
SUPPORTED = (1,)
MAX_SAFE = 9007199254740991
# "a plain file name": ^[A-Za-z0-9][A-Za-z0-9._-]*$ (contract, "The file form").
PLAIN_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# endpoint-is-address pattern, case-insensitive (contract, "Provenance").
ADDRESS = re.compile(r"://|localhost|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|:[0-9]+", re.IGNORECASE)


def _integer(value: object) -> bool:
    """A JSON number with no fractional part, |v| <= 2^53 - 1; ``true`` is not one."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE
    if isinstance(value, float):
        return math.isfinite(value) and value == int(value) and abs(value) <= MAX_SAFE
    return False


def _number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _string(value: object, non_empty: bool = False) -> bool:
    return isinstance(value, str) and (len(value) > 0 or not non_empty)


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _fields(data: dict) -> list[str]:  # noqa: C901 - one branch per row of the tables
    """Every ``invalid-field`` path, in the contract's table order."""
    bad: list[str] = []
    op = data.get("op", "replace")
    if op not in ("replace", "append"):
        bad.append("op")
    if not _string(data.get("model"), non_empty=True):
        bad.append("model")
    dim = data.get("dimension")
    if not (_integer(dim) and dim >= 1):
        bad.append("dimension")
    count = data.get("count")
    if not (_integer(count) and count >= 0):
        bad.append("count")
    if not isinstance(data.get("text_included"), bool):
        bad.append("text_included")

    if op != "append" or "provenance" in data:
        prov = data.get("provenance")
        if not isinstance(prov, dict):
            bad.append("provenance")
        else:
            for name in ("source", "producer"):
                if not _string(prov.get(name), non_empty=True):
                    bad.append(f"provenance.{name}")
            if not isinstance(prov.get("synthetic"), bool):
                bad.append("provenance.synthetic")
            if "seed" not in prov or not (prov["seed"] is None or _integer(prov["seed"])):
                bad.append("provenance.seed")
            for name in ("limitations", "uncertain"):
                if not _strings(prov.get(name)):
                    bad.append(f"provenance.{name}")
            if "created" in prov and not _string(prov["created"]):
                bad.append("provenance.created")
            if "endpoint" in prov and not _string(prov["endpoint"], non_empty=True):
                bad.append("provenance.endpoint")
            if "params" in prov and not isinstance(prov["params"], dict):
                bad.append("provenance.params")

    if "reduction" in data:
        red = data["reduction"]
        if not isinstance(red, dict):
            bad.append("reduction")
        else:
            before = len(bad)
            if red.get("kind") not in ("none", "rotation", "top-k"):
                bad.append("reduction.kind")
            for name in ("source_dimension", "directions_kept"):
                if not (_integer(red.get(name)) and red[name] >= 1):
                    bad.append(f"reduction.{name}")
            dropped = red.get("variance_dropped")
            if not (_number(dropped) and 0 <= dropped <= 1):
                bad.append("reduction.variance_dropped")
            if len(bad) == before and _integer(dim):
                ok = red["directions_kept"] == dim
                if red["kind"] in ("none", "rotation"):
                    ok = ok and red["source_dimension"] == dim and dropped == 0
                else:  # top-k
                    ok = ok and dim < red["source_dimension"] and dropped < 1
                if not ok:
                    bad.append("reduction")

    vectors = data.get("vectors")
    if not isinstance(vectors, dict):
        bad.append("vectors")
    else:
        if not _string(vectors.get("file"), non_empty=True):
            bad.append("vectors.file")
        if vectors.get("dtype") != "float32":
            bad.append("vectors.dtype")
        if vectors.get("byte_order") != "little":
            bad.append("vectors.byte_order")
        want = count * dim * 4 if _integer(count) and _integer(dim) else None
        if not (_integer(vectors.get("byte_length")) and vectors["byte_length"] == want):
            bad.append("vectors.byte_length")

    points = data.get("points")
    if not isinstance(points, list):
        bad.append("points")
    else:
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                bad.append(f"points[{index}]")
                continue
            if not _string(point.get("id"), non_empty=True):
                bad.append(f"points[{index}].id")
            if not _string(point.get("label")):
                bad.append(f"points[{index}].label")
            if "group" in point and not _string(point["group"]):
                bad.append(f"points[{index}].group")
            if "metadata" in point and not isinstance(point["metadata"], dict):
                bad.append(f"points[{index}].metadata")
    return bad


def check(data: object, block: bytes | None) -> list[tuple[str, str]]:
    """Faults of a file-form set, in check order (1 to 10). ``block`` None = unreadable."""
    faults: list[tuple[str, str]] = []
    if not isinstance(data, dict) or data.get("kind") != KIND:
        return [("wrong-kind", "")]
    version = data.get("schema_version")
    if not (_integer(version) and int(version) in SUPPORTED):
        return [("schema-version", json.dumps(version))]
    bad = _fields(data)
    if bad:
        return [("invalid-field", path) for path in bad]

    name = data["vectors"]["file"]
    plain = PLAIN_FILE.match(name) is not None
    if not plain:
        faults.append(("vector-file-path", name))
    endpoint = data.get("provenance", {}).get("endpoint")
    if endpoint is not None and ADDRESS.search(endpoint):
        faults.append(("endpoint-is-address", endpoint))
    points = data["points"]
    if len(points) != data["count"]:
        faults.append(("count-mismatch", str(len(points))))
    seen: set[str] = set()
    for point in points:
        if point["id"] in seen:
            faults.append(("duplicate-id", point["id"]))
        seen.add(point["id"])
    if not plain:
        return faults
    if block is None:
        faults.append(("block-unreadable", name))
        return faults
    count, dim = data["count"], data["dimension"]
    if len(block) != count * dim * 4:
        faults.append(("block-length", str(len(block))))
        return faults
    if len(points) == count:
        values = struct.unpack(f"<{count * dim}f", block)
        for row in range(count):
            if not all(math.isfinite(v) for v in values[row * dim : (row + 1) * dim]):
                faults.append(("non-finite", points[row]["id"]))
    return faults


def check_pair(json_path: Path) -> list[tuple[str, str]]:
    """Load ``<name>.json`` and the block it names beside it, then :func:`check` them."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    block = None
    vectors = data.get("vectors") if isinstance(data, dict) else None
    if isinstance(vectors, dict) and isinstance(vectors.get("file"), str):
        candidate = Path(json_path).parent / vectors["file"]
        if PLAIN_FILE.match(vectors["file"]) and candidate.is_file():
            block = candidate.read_bytes()
    return check(data, block)
