"""Fixed facts about the simulated shop: services, dependency graph, incident types, actions."""

from __future__ import annotations

SERVICES: tuple[str, ...] = (
    "api-gateway",
    "auth",
    "catalog",
    "checkout",
    "payment-db",
    "cache",
)

# (caller, callee): caller depends on callee.
DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("api-gateway", "auth"),
    ("api-gateway", "catalog"),
    ("api-gateway", "checkout"),
    ("catalog", "cache"),
    ("catalog", "payment-db"),
    ("checkout", "auth"),
    ("checkout", "cache"),
    ("checkout", "payment-db"),
)

INCIDENT_TYPES: tuple[str, ...] = (
    "db_pool_exhausted",
    "bad_deploy",
    "memory_leak",
    "cross_zone_latency",
    "expired_certificate",
    "cache_eviction_storm",
)

# One correct action per incident type (the spec's "raise pool limit, restart checkout" etc.).
ACTIONS: dict[str, str] = {
    "db_pool_exhausted": "raise_pool_limit_restart_checkout",
    "bad_deploy": "rollback_deploy",
    "memory_leak": "restart_service_open_bug",
    "cross_zone_latency": "failover_single_zone",
    "expired_certificate": "renew_certificate",
    "cache_eviction_storm": "warm_cache_rate_limit",
}

# Services that may be the true root cause for each incident type.
ROOT_CAUSE_SERVICES: dict[str, tuple[str, ...]] = {
    "db_pool_exhausted": ("checkout",),
    "bad_deploy": ("catalog", "checkout", "auth"),
    "memory_leak": ("catalog", "checkout", "auth"),
    "cross_zone_latency": ("checkout",),
    "expired_certificate": ("auth", "catalog", "payment-db"),
    "cache_eviction_storm": ("cache",),
}

SENSITIVE_TYPES: tuple[str, ...] = ("email", "ip", "api_key", "order_id")

MEMORY_LIMIT_MB: dict[str, int] = {
    "api-gateway": 1024,
    "auth": 1024,
    "catalog": 2048,
    "checkout": 2048,
    "payment-db": 4096,
    "cache": 4096,
}
DB_POOL_MAX = 50
DB_MAX_CONNECTIONS = 100


# Fault variants and their ground-truth labels (where the fault is injected = where the first
# error occurs). Kept here so the pipeline, faults and analysis share one definition.
VARIANTS: tuple[str, ...] = ("E1", "E2", "E3", "R1", "R2", "I1", "I2", "I3")
VARIANT_NAMES: dict[str, str] = {
    "E1": "stale_window",
    "E2": "wrong_service",
    "E3": "missing_alert",
    "R1": "misleading_prior",
    "R2": "unstable_sampling",
    "I1": "dropped_field",
    "I2": "truncated_findings",
    "I3": "dropped_target",
}
FIRST_ERROR_STEP: dict[str, str] = {
    "E1": "evidence_agent",
    "E2": "evidence_agent",
    "E3": "evidence_agent",
    "R1": "reasoning_agent",
    "R2": "reasoning_agent",
    "I1": "handover_1",
    "I2": "handover_1",
    "I3": "handover_2",
}
