"""Seeded generator for simulated online-shop incidents.

One call to ``generate_scenario(seed)`` is a pure function of the seed: it draws every random
choice from a private ``random.Random(seed)``, so the same seed always gives the same scenario.

Construction order (this order matters for determinism):
  1. healthy one-hour baseline for every service (60 one-minute points);
  2. pick the incident type, root-cause service and onset minute;
  3. apply that incident type's signature to the metrics and queue its alerts/logs/deploys;
  4. add decoys (benign deploys, noise logs, an unrelated low-severity alert);
  5. sort everything by time, assign ids, and render text -- planting sensitive values as we go
     and recording exactly where each one landed.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .models import (
    Alert,
    Deploy,
    GroundTruth,
    LogLine,
    MetricStore,
    Scenario,
    SensitiveField,
    Topology,
)
from .taxonomy import (
    ACTIONS,
    DB_MAX_CONNECTIONS,
    DB_POOL_MAX,
    DEPENDENCIES,
    INCIDENT_TYPES,
    MEMORY_LIMIT_MB,
    ROOT_CAUSE_SERVICES,
    SERVICES,
)

N_POINTS = 60
BASE_TIME = datetime(2025, 3, 10, 9, 0, 0, tzinfo=UTC)

_LATENCY_BASE = {
    "api-gateway": 120.0,
    "auth": 40.0,
    "catalog": 60.0,
    "checkout": 150.0,
    "payment-db": 15.0,
    "cache": 3.0,
}
_FIRST = ["ana", "ben", "chloe", "dev", "eli", "fatima", "gus", "hana", "ivan", "jo", "kofi", "lea"]
_LAST = ["lopez", "kim", "nguyen", "okafor", "patel", "rossi", "silva", "tan", "weber", "young"]
_DOMAINS = ["mailbox.test", "shopmail.test", "inbox.test", "example.test"]
_EXCEPTIONS = [
    "NullPointerException in PriceCalculator.apply",
    "IllegalStateException: cart totals mismatch",
    "SerializationException: unknown field 'discountCode'",
    "ValidationError: schema v3 rejected request body",
]
_BENIGN_CHANGES = [
    "bump base image to patched release",
    "update default feature-flag values",
    "reduce log verbosity for health checks",
    "dependency patch-level upgrades",
]
_BAD_CHANGES = [
    "refactor price rounding and request validation",
    "switch JSON serializer library",
    "enable new retry policy for downstream calls",
    "tighten input schema validation",
]

_PLACEHOLDER = re.compile(r"<<(email|ip|api_key|order_id)>>")

# Noise templates per service. The first entries are always emitted so that every scenario
# plants every sensitive type at least once (gateway: ip+api_key, auth: email+ip, checkout:
# order_id+email).
_NOISE: dict[str, list[str]] = {
    "api-gateway": [
        "GET /api/v1/products 200 {ms}ms client_ip=<<ip>>",
        "POST /api/v1/orders 201 {ms}ms client_ip=<<ip>> api_key=<<api_key>>",
        "GET /api/v1/products/{n} 200 {ms}ms",
        "GET /healthz 200 1ms",
    ],
    "auth": [
        "login ok for <<email>> from <<ip>>",
        "api key <<api_key>> validated, scope=orders:write",
        "token refreshed ttl=3600s",
        "session store compaction finished",
    ],
    "catalog": [
        "search q='running shoes' results=24 took={ms}ms",
        "index refresh complete docs={n}",
        "GET /catalog/items/{n} 200 {ms}ms",
    ],
    "checkout": [
        "order <<order_id>> created for <<email>> total=${n}.99",
        "payment authorised for order <<order_id>>",
        "cart session {n} expired, items released",
        "inventory reservation confirmed for order <<order_id>>",
    ],
    "payment-db": [
        "checkpoint complete: wrote {n} buffers",
        "autovacuum finished on table orders",
        "replication lag 0.{ms}s",
    ],
    "cache": [
        "keyspace stats hits={n} misses={ms}",
        "background save finished",
        "replica sync ok",
    ],
}
_GUARANTEED = {"api-gateway": 2, "auth": 2, "checkout": 2}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _noisy(rng: random.Random, mean: float, sd: float) -> list[float]:
    return [max(0.0, rng.gauss(mean, sd)) for _ in range(N_POINTS)]


def _ramp(t0: int, length: int) -> list[float]:
    """0 before onset, rising linearly to 1 over ``length`` minutes, then 1."""
    return [_clamp01((i - t0 + 1) / length) if i >= t0 else 0.0 for i in range(N_POINTS)]


def _ancestors(service: str) -> dict[str, int]:
    """Services that (transitively) depend on ``service``, with hop distance."""
    depth: dict[str, int] = {}
    frontier = [service]
    d = 0
    while frontier:
        d += 1
        nxt: list[str] = []
        for node in frontier:
            for caller, callee in DEPENDENCIES:
                if callee == node and caller not in depth and caller != service:
                    depth[caller] = d
                    nxt.append(caller)
        frontier = nxt
    return depth


@dataclass
class _Draft:
    """Everything queued up before sorting and text rendering."""

    t0: int
    root: str
    symptom: str
    series: dict[str, dict[str, list[float]]]
    zones: dict[str, str]
    logs: list[tuple[int, int, str, str, str]] = field(default_factory=list)
    alerts: list[tuple[int, str, str, str, str, bool]] = field(default_factory=list)
    deploys: list[tuple[int, str, str, str, str]] = field(default_factory=list)

    def log(self, rng: random.Random, minute: int, service: str, level: str, text: str) -> None:
        self.logs.append((min(minute, N_POINTS - 1), rng.randrange(60), service, level, text))

    def alert(
        self, minute: int, service: str, severity: str, name: str, text: str, decisive: bool = False
    ) -> None:
        self.alerts.append((min(minute, N_POINTS - 1), service, severity, name, text, decisive))

    def propagate(
        self, metric: str, root: str, peak: float, shape: list[float], decay: float
    ) -> None:
        for svc, hops in _ancestors(root).items():
            arr = self.series[svc][metric]
            for i, f in enumerate(shape):
                arr[i] += peak * f * decay**hops


class _Planter:
    """Mints synthetic sensitive values and records where each one is placed."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.fields: list[SensitiveField] = []

    def _value(self, kind: str) -> str:
        r = self.rng
        if kind == "email":
            return f"{r.choice(_FIRST)}.{r.choice(_LAST)}{r.randrange(10, 99)}@{r.choice(_DOMAINS)}"
        if kind == "ip":
            first = r.choice([n for n in range(11, 224) if n not in (10, 127, 169, 172, 192)])
            return f"{first}.{r.randrange(256)}.{r.randrange(256)}.{r.randrange(1, 255)}"
        if kind == "api_key":
            alphabet = string.ascii_lowercase + string.digits
            return "sk_live_" + "".join(r.choice(alphabet) for _ in range(24))
        return f"ORD-{r.randrange(10**7, 10**8)}"

    def render(self, template: str, source: str, index: int, field_name: str) -> str:
        out: list[str] = []
        pos = last = 0
        for m in _PLACEHOLDER.finditer(template):
            chunk = template[last : m.start()]
            out.append(chunk)
            pos += len(chunk)
            value = self._value(m.group(1))
            self.fields.append(
                SensitiveField(
                    id=f"S{len(self.fields) + 1}",
                    type=m.group(1),
                    value=value,
                    source=source,
                    index=index,
                    field=field_name,
                    start=pos,
                    end=pos + len(value),
                )
            )
            out.append(value)
            pos += len(value)
            last = m.end()
        out.append(template[last:])
        return "".join(out)


def _ts(minute: int, second: int = 0) -> str:
    return (BASE_TIME + timedelta(minutes=minute, seconds=second)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _baseline(rng: random.Random) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    for svc in SERVICES:
        lat = _LATENCY_BASE[svc] * rng.uniform(0.9, 1.1)
        mem_limit = MEMORY_LIMIT_MB[svc]
        series[svc] = {
            "latency_p95_ms": _noisy(rng, lat, lat * 0.05),
            "error_rate": _noisy(rng, 0.002, 0.0006),
            "cpu_pct": _noisy(rng, rng.uniform(25, 45), 2.0),
            "memory_mb": _noisy(rng, mem_limit * rng.uniform(0.4, 0.5), mem_limit * 0.004),
            "tls_handshake_failures": [0.0] * N_POINTS,
        }
        days_left = rng.uniform(20, 90) * 86400
        series[svc]["cert_seconds_to_expiry"] = [days_left - 60 * i for i in range(N_POINTS)]
    series["checkout"]["db_pool_in_use"] = _noisy(rng, rng.uniform(10, 18), 1.5)
    series["checkout"]["payment_db_rpc_latency_ms"] = _noisy(rng, 12.0, 1.0)
    series["payment-db"]["active_connections"] = _noisy(rng, rng.uniform(20, 30), 2.0)
    series["cache"]["hit_ratio"] = _noisy(rng, 0.93, 0.01)
    series["cache"]["evictions_per_s"] = _noisy(rng, 2.0, 0.5)
    return series


def _add(arr: list[float], peak: float, shape: list[float]) -> None:
    for i, f in enumerate(shape):
        arr[i] += peak * f


# ---- incident signatures ---------------------------------------------------------------------


def _db_pool_exhausted(rng: random.Random, d: _Draft) -> None:
    f = _ramp(d.t0, 4)
    pool = d.series["checkout"]["db_pool_in_use"]
    for i in range(N_POINTS):
        pool[i] = pool[i] + (DB_POOL_MAX - pool[i]) * f[i]
        if f[i] >= 1.0:
            pool[i] = DB_POOL_MAX - rng.choice([0, 0, 1])
    conns = d.series["payment-db"]["active_connections"]
    target = rng.uniform(88, 97)
    for i in range(N_POINTS):
        conns[i] += (target - conns[i]) * f[i]
    _add(d.series["checkout"]["latency_p95_ms"], rng.uniform(900, 2500), f)
    _add(d.series["checkout"]["error_rate"], rng.uniform(0.12, 0.35), f)
    d.propagate("latency_p95_ms", "checkout", 700.0, f, 0.7)
    d.propagate("error_rate", "checkout", 0.1, f, 0.5)
    for k in range(3):
        d.log(
            rng,
            d.t0 + k,
            "checkout",
            "ERROR",
            f"timeout acquiring connection from pool (pool_size={DB_POOL_MAX}, "
            f"in_use={DB_POOL_MAX}, waited_ms={rng.randrange(5000, 30000)})",
        )
    d.log(rng, d.t0 + 2, "checkout", "WARN", f"db pool wait queue length={rng.randrange(40, 220)}")
    d.log(
        rng,
        d.t0 + 3,
        "payment-db",
        "WARN",
        f"connection count {int(target)}/{DB_MAX_CONNECTIONS}, approaching max_connections",
    )
    d.log(
        rng,
        d.t0 + 3,
        d.symptom,
        "ERROR",
        "upstream checkout returned 504 for POST /api/v1/orders client_ip=<<ip>>",
    )
    peak = DB_POOL_MAX
    d.alert(
        d.t0 + rng.randint(1, 3),
        "checkout",
        "critical",
        "DbPoolSaturation",
        f"db_pool_in_use at {peak}/{DB_POOL_MAX} (>=95% of pool limit)",
        decisive=True,
    )


def _bad_deploy(rng: random.Random, d: _Draft) -> None:
    f = _ramp(d.t0, 1)
    root = d.root
    version = f"v{rng.randint(1, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
    deploy_min = d.t0 - rng.randint(1, 2)
    d.deploys.append((deploy_min, root, "deploy", version, rng.choice(_BAD_CHANGES)))
    step = rng.uniform(0.08, 0.25)
    _add(d.series[root]["error_rate"], step, f)
    _add(d.series[root]["latency_p95_ms"], rng.uniform(150, 600), f)
    d.propagate("latency_p95_ms", root, 300.0, f, 0.7)
    d.propagate("error_rate", root, step * 0.6, f, 0.5)
    for k in range(3):
        d.log(
            rng,
            d.t0 + k,
            root,
            "ERROR",
            f"unhandled exception in request handler (build {version}): {rng.choice(_EXCEPTIONS)}",
        )
    d.log(
        rng,
        d.t0 + 1,
        d.symptom,
        "ERROR",
        f"upstream {root} returned 500 for GET /api/v1/orders client_ip=<<ip>>",
    )
    d.alert(
        d.t0 + rng.randint(1, 3),
        root,
        "critical",
        "ErrorRateStepChange",
        f"error rate on {root} stepped from 0.2% to {step * 100 + 0.2:.1f}% right after "
        f"rollout of {version}",
        decisive=True,
    )


def _memory_leak(rng: random.Random, d: _Draft) -> None:
    root = d.root
    limit = MEMORY_LIMIT_MB[root]
    mem = d.series[root]["memory_mb"]
    for i in range(N_POINTS):
        frac = 0.45 + 0.52 * i / (N_POINTS - 1)
        mem[i] = limit * frac * (1 + rng.gauss(0, 0.004))
    f = _ramp(d.t0, 15)
    _add(d.series[root]["latency_p95_ms"], rng.uniform(100, 400), f)
    _add(d.series[root]["error_rate"], rng.uniform(0.005, 0.02), f)
    d.propagate("latency_p95_ms", root, 200.0, f, 0.7)
    for k in range(3):
        d.log(
            rng,
            d.t0 + 4 * k + 2,
            root,
            "WARN",
            f"heap usage {70 + 8 * k + rng.randint(0, 3)}% of {limit}MB limit, "
            f"GC pause {rng.randrange(400, 2500)}ms",
        )
    d.log(
        rng, N_POINTS - 3, root, "ERROR", f"allocation failure: approaching OOM at {limit}MB limit"
    )
    d.alert(
        d.t0 + rng.randint(6, 14),
        root,
        "warning",
        "MemoryNearLimit",
        f"memory at 9{rng.randint(0, 6)}% of {limit}MB limit and still rising",
        decisive=True,
    )


def _cross_zone_latency(rng: random.Random, d: _Draft) -> None:
    f = _ramp(d.t0, 2)
    d.zones["checkout"] = "zone-b"
    d.deploys.append(
        (
            d.t0,
            "checkout",
            "infra",
            "n/a",
            "autoscaler rescheduled checkout pods from zone-a to zone-b "
            "(payment-db remains in zone-a)",
        )
    )
    rpc = rng.uniform(35, 80)
    _add(d.series["checkout"]["payment_db_rpc_latency_ms"], rpc, f)
    _add(d.series["checkout"]["latency_p95_ms"], rpc * rng.uniform(2.5, 4.0), f)
    _add(d.series["checkout"]["error_rate"], rng.uniform(0.001, 0.004), f)
    d.propagate("latency_p95_ms", "checkout", rpc * 2.5, f, 0.7)
    d.log(rng, d.t0, "checkout", "INFO", "pod rescheduled to zone-b by autoscaler")
    for k in range(3):
        d.log(
            rng,
            d.t0 + 2 + k,
            "checkout",
            "WARN",
            f"payment-db call took {int(12 + rpc)}ms (src zone=zone-b dst zone=zone-a)",
        )
    d.alert(
        d.t0 + rng.randint(2, 4),
        "checkout",
        "warning",
        "CrossZoneRpcLatency",
        f"checkout->payment-db rpc p95 {12 + rpc:.0f}ms (baseline 12ms)",
        decisive=True,
    )


def _expired_certificate(rng: random.Random, d: _Draft) -> None:
    root = d.root
    f = _ramp(d.t0, 1)
    cert = d.series[root]["cert_seconds_to_expiry"]
    for i in range(N_POINTS):
        cert[i] = (d.t0 - i) * 60 - rng.randint(0, 30)
    fails = rng.uniform(40, 150)
    _add(d.series[root]["tls_handshake_failures"], fails, f)
    err = rng.uniform(0.55, 0.95)
    _add(d.series[root]["error_rate"], err, f)
    d.propagate("error_rate", root, err, f, 0.8)
    d.propagate("latency_p95_ms", root, rng.uniform(50, 200), f, 0.7)
    d.log(
        rng,
        d.t0,
        root,
        "ERROR",
        "tls handshake failed: x509: certificate has expired or is not yet valid",
    )
    for k, caller in enumerate(sorted(_ancestors(root))[:2]):
        d.log(
            rng,
            d.t0 + 1 + k,
            caller,
            "ERROR",
            f"call to {root} failed: tls: failed to verify certificate: "
            "x509: certificate has expired",
        )
    d.alert(
        d.t0 + rng.randint(1, 2),
        root,
        "critical",
        "TlsCertificateExpired",
        f"certificate for {root} expired; handshake failures {fails:.0f}/min",
        decisive=True,
    )


def _cache_eviction_storm(rng: random.Random, d: _Draft) -> None:
    f = _ramp(d.t0, 3)
    storm = rng.uniform(600, 2400)
    _add(d.series["cache"]["evictions_per_s"], storm, f)
    _add(d.series["cache"]["hit_ratio"], -rng.uniform(0.35, 0.55), f)
    _add(d.series["cache"]["latency_p95_ms"], rng.uniform(15, 40), f)
    d.propagate("latency_p95_ms", "cache", rng.uniform(250, 700), f, 0.8)
    _add(d.series["payment-db"]["active_connections"], rng.uniform(30, 42), f)
    _add(d.series["checkout"]["db_pool_in_use"], rng.uniform(8, 14), f)
    _add(d.series["catalog"]["error_rate"], rng.uniform(0.02, 0.06), f)
    d.propagate("error_rate", "catalog", 0.03, f, 0.5)
    for k in range(3):
        d.log(
            rng,
            d.t0 + k,
            "cache",
            "WARN",
            f"evicted {int(storm * rng.uniform(0.8, 1.2))} keys in 1s "
            "(maxmemory-policy=allkeys-lru)",
        )
    d.log(
        rng,
        d.t0 + 2,
        "catalog",
        "WARN",
        f"cache miss ratio {rng.randint(45, 70)}%, falling back to payment-db",
    )
    d.alert(
        d.t0 + rng.randint(1, 3),
        "cache",
        "critical",
        "CacheEvictionRate",
        f"evictions {storm:.0f}/s (baseline 2/s), hit ratio below 60%",
        decisive=True,
    )


_APPLY = {
    "db_pool_exhausted": _db_pool_exhausted,
    "bad_deploy": _bad_deploy,
    "memory_leak": _memory_leak,
    "cross_zone_latency": _cross_zone_latency,
    "expired_certificate": _expired_certificate,
    "cache_eviction_storm": _cache_eviction_storm,
}


# ---- assembly --------------------------------------------------------------------------------


def _add_decoys(rng: random.Random, d: _Draft, incident_type: str) -> None:
    """Benign deploys (>=12 min from onset), noise logs, and one unrelated low-severity alert."""
    others = [s for s in SERVICES if s != d.root]
    n_benign = rng.randint(1, 2)
    for _ in range(n_benign):
        svc = rng.choice(others)
        minute = rng.randrange(0, d.t0 - 12)
        v = f"v{rng.randint(1, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
        d.deploys.append((minute, svc, "deploy", v, rng.choice(_BENIGN_CHANGES)))
    for svc in SERVICES:
        templates = _NOISE[svc]
        guaranteed = _GUARANTEED.get(svc, 0)
        picks = list(range(guaranteed)) + [
            rng.randrange(len(templates)) for _ in range(rng.randint(3, 6))
        ]
        for idx in picks:
            text = templates[idx].format(ms=rng.randrange(3, 90), n=rng.randrange(100, 9999))
            d.log(rng, rng.randrange(N_POINTS), svc, "INFO", text)
        if rng.random() < 0.4:
            d.log(
                rng,
                rng.randrange(N_POINTS),
                svc,
                "WARN",
                f"slow request {rng.randrange(200, 450)}ms",
            )
    noisy = rng.choice(others)
    d.alert(
        rng.randrange(N_POINTS),
        noisy,
        "info",
        "DiskUsageNotice",
        f"disk usage on {noisy} at {rng.randint(61, 74)}%",
    )


def _symptom_service(rng: random.Random, root: str) -> str:
    if root in ("checkout", "catalog", "auth"):
        return "api-gateway"
    return rng.choice(sorted(c for c, callee in DEPENDENCIES if callee == root))


def generate_scenario(seed: int, incident_type: str | None = None) -> Scenario:
    """Build one scenario; with no ``incident_type`` it is ``INCIDENT_TYPES[seed % 6]``."""
    rng = random.Random(seed)
    if incident_type is None:
        incident_type = INCIDENT_TYPES[seed % len(INCIDENT_TYPES)]
    if incident_type not in _APPLY:
        raise ValueError(f"unknown incident type: {incident_type}")

    root = rng.choice(ROOT_CAUSE_SERVICES[incident_type])
    draft = _Draft(
        t0=rng.randint(28, 40),
        root=root,
        symptom=_symptom_service(rng, root),
        series=_baseline(rng),
        zones={s: "zone-a" for s in SERVICES},
    )
    _APPLY[incident_type](rng, draft)

    # generic user-visible symptom alerts
    peak_gw = max(draft.series["api-gateway"]["latency_p95_ms"])
    draft.alert(
        draft.t0 + rng.randint(2, 5),
        "api-gateway",
        "critical",
        "HighLatencyP95",
        f"gateway p95 latency {peak_gw:.0f}ms (baseline ~120ms); sample client_ip=<<ip>>",
    )
    if draft.symptom != "api-gateway":
        draft.alert(
            draft.t0 + rng.randint(2, 5),
            draft.symptom,
            "warning",
            "HighErrorRate",
            f"error rate elevated on {draft.symptom}",
        )
    if "checkout" in (root, draft.symptom) and rng.random() < 0.5:
        draft.alert(
            draft.t0 + rng.randint(3, 6),
            "checkout",
            "warning",
            "FailedOrders",
            "order placement failures rising, sample order <<order_id>>",
        )

    _add_decoys(rng, draft, incident_type)

    planter = _Planter(rng)

    deploys_sorted = sorted(enumerate(draft.deploys), key=lambda p: (p[1][0], p[0]))
    deploys: list[Deploy] = []
    for idx, (_, (minute, svc, kind, version, desc)) in enumerate(deploys_sorted):
        author = planter.render("<<email>>", "deploys", idx, "author")
        deploys.append(
            Deploy(
                id=f"D{idx + 1}",
                time=_ts(minute),
                service=svc,
                kind=kind,
                version=version,
                author=author,
                description=desc,
            )
        )

    alerts_sorted = sorted(enumerate(draft.alerts), key=lambda p: (p[1][0], p[0]))
    alerts: list[Alert] = []
    decisive_id = ""
    for idx, (_, (minute, svc, sev, name, text, decisive)) in enumerate(alerts_sorted):
        aid = f"A{idx + 1}"
        if decisive:
            decisive_id = aid
        alerts.append(
            Alert(
                id=aid,
                time=_ts(minute, 0),
                service=svc,
                severity=sev,
                name=name,
                message=planter.render(text, "alerts", idx, "message"),
            )
        )

    logs_sorted = sorted(enumerate(draft.logs), key=lambda p: (p[1][0], p[1][1], p[0]))
    logs: list[LogLine] = []
    for idx, (_, (minute, sec, svc, level, text)) in enumerate(logs_sorted):
        logs.append(
            LogLine(
                time=_ts(minute, sec),
                service=svc,
                level=level,
                message=planter.render(text, "logs", idx, "message"),
            )
        )

    timestamps = [_ts(i) for i in range(N_POINTS)]
    series = {
        svc: {m: [round(v, 5) for v in vals] for m, vals in metrics.items()}
        for svc, metrics in draft.series.items()
    }
    limits: dict[str, float] = {
        "checkout.db_pool_max": float(DB_POOL_MAX),
        "payment-db.max_connections": float(DB_MAX_CONNECTIONS),
    }
    limits.update({f"{s}.memory_limit_mb": float(v) for s, v in MEMORY_LIMIT_MB.items()})

    return Scenario(
        scenario_id=f"sc-{seed:05d}",
        seed=seed,
        incident_type=incident_type,
        time_window=(timestamps[0], timestamps[-1]),
        topology=Topology(
            services=list(SERVICES),
            dependencies=list(DEPENDENCIES),
            zones=draft.zones,
            limits=limits,
        ),
        alerts=alerts,
        metrics=MetricStore(timestamps=timestamps, series=series),
        deploys=deploys,
        logs=logs,
        sensitive_fields=planter.fields,
        ground_truth=GroundTruth(
            incident_type=incident_type,
            root_cause=incident_type,
            root_cause_service=root,
            symptom_service=draft.symptom,
            correct_action=ACTIONS[incident_type],
            action_target=root,
            onset_time=timestamps[draft.t0],
            decisive_alert_id=decisive_id,
        ),
    )


def generate_many(base_seed: int, count: int) -> list[Scenario]:
    """Scenarios for seeds base_seed .. base_seed+count-1; types cycle so the mix is balanced."""
    return [generate_scenario(base_seed + i) for i in range(count)]
