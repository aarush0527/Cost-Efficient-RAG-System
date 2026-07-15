# Architecture overview

NimbusStore is an object storage and backup service. This document describes
how data is stored, replicated, and made durable.

## Regions

NimbusStore operates in three regions: us-east-nimbus-1, us-west-nimbus-2,
and eu-central-nimbus-1. Buckets are created in exactly one region and data
does not automatically move between regions unless the customer configures
cross-region replication.

## Replication and durability

Within a region, every object is replicated 3 times synchronously across
separate availability zones before a write is acknowledged back to the
client. This synchronous replication is what backs the durability target of
99.995% annual durability per object.

After 90 days of inactivity, an object automatically transitions to the cold
storage tier. Cold tier objects are re-encoded using a 6-of-9 erasure coding
scheme instead of 3x replication: the object is split into 9 fragments, any
6 of which are sufficient to reconstruct it. This reduces raw storage
overhead from roughly 3x (replication) to about 1.5x (erasure coding), which
is why cold tier storage is billed at a lower rate - see pricing.md.

## Object size limits

A single object may be up to 5 TB. Objects larger than 100 MB must be
uploaded using the multipart upload API rather than a single PUT request;
each part in a multipart upload may be up to 5 GB.

## Metadata layer

Object location, version history, and access-control metadata are tracked
in a separate distributed key-value index called NimbusIndex, which is
itself replicated independently of the object data it describes.
