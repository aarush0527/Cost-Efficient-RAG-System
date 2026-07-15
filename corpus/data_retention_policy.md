# Data retention policy

## Deleted objects

When an object is deleted, it is soft-deleted first: it is recoverable for
30 days by default. Enterprise customers can configure this recovery window
up to 90 days. After the recovery window elapses, the object is permanently
purged and cannot be recovered by NimbusStore or the customer.

## Backups

Beyond the synchronous replication and erasure coding described in
architecture_overview.md, NimbusStore does not maintain a separate backup
copy of customer objects. Customers who need protection against, for
example, accidental bulk deletion across regions are responsible for their
own additional backup strategy (such as replicating to a second bucket in a
different region).

## Account closure

After an account closure is confirmed, all associated data is permanently
deleted 14 days later. There is no recovery mechanism after that 14-day
window closes, so customers should export any data they want to keep before
confirming closure.

## Audit log retention

Audit logs (see security_compliance.md) are retained for 400 days by
default. Customers may request a shorter retention period for their own
audit logs, but NimbusStore will not reduce it below 90 days under any
circumstance, since that minimum is required for internal security
investigations.
