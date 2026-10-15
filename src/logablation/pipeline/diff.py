"""Field-level diff between the object a handover sent and the object it delivered."""

from __future__ import annotations

from typing import Any


def field_diff(sent: Any, received: Any, path: str = "") -> list[dict[str, Any]]:
    """Every leaf-level difference as {path, op, before, after}; identical objects give [].

    op is "removed" (in sent only), "added" (in received only) or "changed". Dict keys are
    joined with dots, list items are addressed by index ("findings[1].value").
    """
    if isinstance(sent, dict) and isinstance(received, dict):
        out: list[dict[str, Any]] = []
        for k in sent.keys() | received.keys():
            sub = f"{path}.{k}" if path else str(k)
            if k not in received:
                out.append({"path": sub, "op": "removed", "before": sent[k], "after": None})
            elif k not in sent:
                out.append({"path": sub, "op": "added", "before": None, "after": received[k]})
            else:
                out.extend(field_diff(sent[k], received[k], sub))
        return sorted(out, key=lambda d: d["path"])
    if isinstance(sent, (list, tuple)) and isinstance(received, (list, tuple)):
        out = []
        for i in range(max(len(sent), len(received))):
            sub = f"{path}[{i}]"
            if i >= len(received):
                out.append({"path": sub, "op": "removed", "before": sent[i], "after": None})
            elif i >= len(sent):
                out.append({"path": sub, "op": "added", "before": None, "after": received[i]})
            else:
                out.extend(field_diff(sent[i], received[i], sub))
        return out
    if sent != received:
        return [{"path": path, "op": "changed", "before": sent, "after": received}]
    return []
