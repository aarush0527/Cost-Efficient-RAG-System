# Pricing

NimbusStore has four tiers: Free, Starter, Pro, and Enterprise.

## Free tier

5 GB of storage and 10 GB of egress per month, at no cost. Intended for
evaluation and small personal projects.

## Starter tier

Storage is billed at $0.018 per GB-month. Egress (data downloaded out of
NimbusStore) is billed at $0.05 per GB. Requests are billed separately:
$0.004 per 1,000 GET requests and $0.02 per 1,000 PUT requests.

## Pro tier

Storage is billed at $0.015 per GB-month - lower than Starter - and includes
100 GB of egress free every month, with additional egress billed at $0.03
per GB.

## Enterprise tier

Custom pricing negotiated per contract. Volume discounts begin at 500 TB of
total stored data. Enterprise customers also get dedicated support - see
sla_support_policy.pdf for response-time commitments.

## Cold storage tier

Objects that have auto-archived to the cold tier (see architecture_overview.md
for how and when this happens) are billed at $0.004 per GB-month, regardless
of account tier. If a cold-tier object is deleted before it has been in the
cold tier for 180 days, an early-deletion fee applies, prorated to the
remaining days out of that 180-day minimum at the cold tier's per-GB rate.
