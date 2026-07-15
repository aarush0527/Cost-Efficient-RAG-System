"""
Monthly cost projection at 100K / 1M / 10M vectors: self-hosted embedded
Qdrant (this project) vs. managed alternatives.

Every number below is a *stated assumption*, not a hidden constant - change
any of them and re-run to see how the comparison shifts. Pricing figures for
the managed services are cited inline (see eval/results/cost_latency.md for
the source links) and were current as of July 2026; vector-database pricing
changes often, so re-verify before using this for a real budget decision.

Deliberately modeled on OUR configuration (384-dim, since that's the
recommended production embedder - see ragapp/embeddings.py), not a generic
1536-dim OpenAI-style embedding, so the two sides of the comparison are
apples-to-apples with what this project actually stores.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- shared assumptions ---
VECTOR_DIM = 384                # bge-small-en-v1.5 (recommended production embedder)
BYTES_PER_FLOAT = 4              # float32
HNSW_OVERHEAD_FACTOR = 1.5       # graph overhead on top of raw vectors (~1.5x is a commonly cited rule of thumb)
PAYLOAD_BYTES_PER_CHUNK = 1200   # ~1000 chars of chunk text + source_file/section_path/etc metadata

QUERIES_PER_DAY = 1000           # "lightly queried" assumption for this whole comparison
DAYS_PER_MONTH = 30
PINECONE_READ_UNITS_PER_QUERY = 2   # a filtered query typically costs >1 RU; see cited source for the 1-10 RU range
PINECONE_WRITES_PER_MONTH = 1000    # small trickle of updates, not a full re-ingest every month

SCALES = [100_000, 1_000_000, 10_000_000]


def raw_vector_bytes(n_vectors: int) -> float:
    return n_vectors * VECTOR_DIM * BYTES_PER_FLOAT


def total_storage_gb(n_vectors: int, include_hnsw_overhead: bool) -> float:
    vec_bytes = raw_vector_bytes(n_vectors) * (HNSW_OVERHEAD_FACTOR if include_hnsw_overhead else 1.0)
    payload_bytes = n_vectors * PAYLOAD_BYTES_PER_CHUNK
    return (vec_bytes + payload_bytes) / (1024 ** 3)


@dataclass
class CostEstimate:
    label: str
    monthly_usd: float
    notes: str


def self_hosted_embedded_qdrant(n_vectors: int) -> CostEstimate:
    """This project's actual deployment shape: embedded Qdrant, on-disk
    (mmap) storage rather than requiring the whole index resident in RAM -
    a reasonable choice specifically for a *lightly queried* index, trading
    a bit of p95 latency for a compute footprint that doesn't have to scale
    with vector count. Fixed compute dominates at this scale; disk is cheap."""
    storage_gb = total_storage_gb(n_vectors, include_hnsw_overhead=True)
    fixed_vm_usd = 18.0          # small VM (e.g. 2 vCPU / 2GB RAM), constant across all three scales
    disk_usd_per_gb_month = 0.09  # e.g. AWS gp3 / DigitalOcean block storage, ballpark
    disk_usd = storage_gb * disk_usd_per_gb_month
    total = fixed_vm_usd + disk_usd
    return CostEstimate(
        label="Self-hosted embedded Qdrant (this project)",
        monthly_usd=round(total, 2),
        notes=f"{storage_gb:.2f} GB total (vectors+HNSW+payload) @ ${disk_usd_per_gb_month}/GB "
              f"+ ${fixed_vm_usd}/mo fixed small VM. No HA/replication/backup included - see trade-offs.",
    )


def pinecone_serverless_usage_only(n_vectors: int) -> CostEstimate:
    """Pinecone's current (2026) default pricing model: pay for storage +
    read/write units, no idle/always-on charge. Storage $0.33/GB-mo, ~$16/M
    read units, ~$4/M write units (Standard plan, per cited sources)."""
    storage_gb = total_storage_gb(n_vectors, include_hnsw_overhead=False)  # Pinecone bills raw stored bytes
    storage_usd = storage_gb * 0.33
    monthly_queries = QUERIES_PER_DAY * DAYS_PER_MONTH
    read_units = monthly_queries * PINECONE_READ_UNITS_PER_QUERY
    read_usd = (read_units / 1_000_000) * 16.0
    write_usd = (PINECONE_WRITES_PER_MONTH / 1_000_000) * 4.0
    total = storage_usd + read_usd + write_usd
    return CostEstimate(
        label="Pinecone Serverless - raw usage-based cost",
        monthly_usd=round(total, 2),
        notes=f"{storage_gb:.2f} GB @ $0.33/GB storage + {monthly_queries:,} queries @ "
              f"{PINECONE_READ_UNITS_PER_QUERY} RU each + {PINECONE_WRITES_PER_MONTH} writes/mo.",
    )


def pinecone_serverless_realistic(n_vectors: int) -> CostEstimate:
    """Same usage as above, but reflecting the real-world $50/mo minimum on
    Pinecone's paid Standard plan once an index exceeds the free tier
    (~2 GB / ~350K vectors uncompressed) - the free tier itself would cover
    the lower end of our range at $0."""
    storage_gb = total_storage_gb(n_vectors, include_hnsw_overhead=False)
    usage = pinecone_serverless_usage_only(n_vectors)
    FREE_TIER_GB_CEILING = 2.0
    if storage_gb <= FREE_TIER_GB_CEILING:
        return CostEstimate(
            label="Pinecone Serverless - realistic bill",
            monthly_usd=0.0,
            notes=f"{storage_gb:.2f} GB fits Pinecone's free tier (~2GB) - likely $0/mo.",
        )
    total = max(usage.monthly_usd, 50.0)
    return CostEstimate(
        label="Pinecone Serverless - realistic bill",
        monthly_usd=round(total, 2),
        notes=f"exceeds free tier ({storage_gb:.2f} GB) -> Standard plan's $50/mo minimum "
              f"{'is' if total == 50.0 else 'is exceeded by'} the binding cost "
              f"(raw usage was ${usage.monthly_usd}/mo).",
    )


def pinecone_legacy_pods(n_vectors: int) -> CostEstimate:
    """The classic "always-on pods" model the assignment's background
    describes - officially legacy in 2026 (new users default to serverless)
    but still available, and the clearest illustration of "you pay for
    capacity, not usage". A single s1 pod (~$70/mo) is roughly sized for a
    few million vectors at moderate dimensionality; we do not have a
    precise, current, officially-published pod-capacity table, so we model
    this coarsely as 1 pod per <=5M vectors - treat as illustrative, not exact."""
    pods_needed = max(1, -(-n_vectors // 5_000_000))  # ceiling division
    per_pod_usd = 70.0
    total = pods_needed * per_pod_usd
    return CostEstimate(
        label="Pinecone legacy dedicated pods (illustrative)",
        monthly_usd=round(total, 2),
        notes=f"~{pods_needed} s1.x1 pod(s) @ ~${per_pod_usd}/mo each, billed whether queried or not "
              f"- this is the specific model the assignment's background describes.",
    )


def qdrant_cloud_managed(n_vectors: int) -> CostEstimate:
    """Qdrant Cloud doesn't publish flat per-unit rates (pricing flows
    through their calculator based on allocated vCPU/RAM/disk), so this is
    interpolated from third-party production-cluster estimates rather than
    an official rate card - see cited sources. Free tier (1GB RAM/4GB disk)
    covers small deployments; we scale linearly by storage above that."""
    storage_gb = total_storage_gb(n_vectors, include_hnsw_overhead=True)
    FREE_TIER_DISK_GB = 4.0
    if storage_gb <= FREE_TIER_DISK_GB:
        return CostEstimate(
            label="Qdrant Cloud (managed, same engine)",
            monthly_usd=0.0,
            notes=f"{storage_gb:.2f} GB fits the permanent free tier (1GB RAM / 4GB disk) - $0/mo.",
        )
    # anchor points from third-party estimates (see cost_latency.md): ~$65-130/mo for
    # a 5M-vector/768-dim production 3-node cluster; scale roughly by our storage footprint
    anchor_gb = total_storage_gb(5_000_000, include_hnsw_overhead=True) * (768 / VECTOR_DIM)
    anchor_usd = 97.5  # midpoint of the $65-130 range
    scaled = anchor_usd * (storage_gb / anchor_gb)
    total = max(30.0, scaled)  # $30/mo Standard tier floor, per cited source
    return CostEstimate(
        label="Qdrant Cloud (managed, same engine)",
        monthly_usd=round(total, 2),
        notes=f"{storage_gb:.2f} GB, scaled from third-party production-cluster estimates "
              f"(~$65-130/mo at 5M vectors/768-dim) - not an official flat rate.",
    )


def build_comparison_table() -> list[dict]:
    rows = []
    for n in SCALES:
        row = {
            "n_vectors": n,
            "self_hosted": self_hosted_embedded_qdrant(n),
            "pinecone_usage": pinecone_serverless_usage_only(n),
            "pinecone_realistic": pinecone_serverless_realistic(n),
            "pinecone_legacy_pods": pinecone_legacy_pods(n),
            "qdrant_cloud": qdrant_cloud_managed(n),
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    for row in build_comparison_table():
        print(f"\n=== {row['n_vectors']:,} vectors ===")
        for key in ("self_hosted", "pinecone_usage", "pinecone_realistic", "pinecone_legacy_pods", "qdrant_cloud"):
            e = row[key]
            print(f"  {e.label}: ${e.monthly_usd}/mo  ({e.notes})")
