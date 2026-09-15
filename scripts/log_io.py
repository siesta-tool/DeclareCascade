"""Minimal XES reader: stream a (possibly gzipped) log into activity-name tuples.

Template/metric computation lives in batch_support_series, not here.
"""

from __future__ import annotations

import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ATTRIBUTE_TAGS = {"string", "date", "int", "float", "boolean"}


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def open_xes(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def parse_traces(path: Path, activity_key: str) -> list[tuple[str, ...]]:
    """Stream an XES file into a list of traces (each a tuple of activity names)."""
    traces: list[tuple[str, ...]] = []
    current_trace: list[str] | None = None
    current_event_attrs: dict[str, str] | None = None
    root: ET.Element | None = None
    warned_substitution = False
    n_missing = 0

    with open_xes(path) as fh:
        try:
            for event, elem in ET.iterparse(fh, events=("start", "end")):
                if root is None:
                    root = elem
                tag = local_name(elem.tag)

                if event == "start":
                    if tag == "trace":
                        current_trace = []
                    elif tag == "event":
                        current_event_attrs = {}
                    continue

                if tag in ATTRIBUTE_TAGS and current_event_attrs is not None:
                    key = elem.attrib.get("key", "")
                    if key:
                        current_event_attrs[key] = elem.attrib.get("value", "")
                    elem.clear()
                    continue

                if tag == "event" and current_event_attrs is not None:
                    # Resolve by membership, not truthiness: an empty activity name is a real
                    # value, and a mistyped activity_key must not resolve silently to another
                    # attribute.
                    for key in (activity_key, "concept:name", "Activity"):
                        if key in current_event_attrs:
                            activity = current_event_attrs[key]
                            if key != activity_key and not warned_substitution:
                                warned_substitution = True
                                print(
                                    f"[warn] {path.name}: no '{activity_key}' attribute; using '{key}'",
                                    file=sys.stderr,
                                )
                            break
                    else:
                        activity = "<missing>"
                        n_missing += 1
                    if current_trace is not None:
                        current_trace.append(activity)
                    current_event_attrs = None
                    elem.clear()
                    continue

                if tag == "trace" and current_trace is not None:
                    traces.append(tuple(current_trace))
                    current_trace = None
                    elem.clear()
                    # Completed traces stay children of <log> otherwise, so elem.clear() alone
                    # leaves retention O(n_traces).
                    del root[:]
        except ET.ParseError as exc:
            raise SystemExit(f"malformed XES: {path}: {exc}") from exc

    if n_missing:
        print(f"[warn] {path.name}: {n_missing} events lacked an activity key", file=sys.stderr)
    return traces


def log_base_name(log_path: Path) -> str:
    name = log_path.name
    if name.endswith(".xes.gz"):
        name = name[:-7]
    elif name.endswith(".xes"):
        name = name[:-4]
    return name
