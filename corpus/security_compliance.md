# Security and compliance

## Encryption

All objects are encrypted at rest using AES-256 by default, with keys
managed by NimbusStore. Customer-managed keys (bring-your-own-key, or BYOK)
are available on the Pro and Enterprise tiers for customers who need to
control their own key material.

All traffic to the API and web console requires TLS 1.3; older TLS versions
are rejected at the load balancer.

## Access control

Access is governed by NimbusIAM: roles and per-bucket policies following a
least-privilege model, similar in spirit to common cloud IAM systems. Root
account access requires multi-factor authentication (MFA). MFA is optional
but strongly recommended for sub-accounts.

## Audit logging

Every read, write, and delete operation is recorded by NimbusAudit into a
separate, immutable log bucket that the acting account cannot modify or
delete from. Audit logs are retained for 400 days by default.

## Compliance

A SOC 2 Type II report is available on request to customers on the Pro and
Enterprise tiers, under NDA. ISO 27001 certification is currently in
progress, targeted for completion in Q3. Starter and Free tier customers do
not currently have access to compliance reports.
