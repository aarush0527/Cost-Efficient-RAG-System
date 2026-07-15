# Troubleshooting FAQ

## I'm getting 403 errors

The most common cause is an expired bearer token - tokens expire 12 hours
after issuance. Generate a new token from the NimbusIAM console. The second
most common cause is a bucket policy that doesn't grant your role the
permission you're trying to use.

## I'm getting 429 errors

You've exceeded the default rate limit of 200 requests per second for your
API token. Back off and retry using the number of seconds given in the
response's `Retry-After` header. Enterprise customers who need a higher
sustained rate should contact their account manager.

## My upload fails for files over 100 MB

Single-request uploads (`PUT /buckets/{bucket}/objects/{key}`) only support
objects up to 100 MB. Anything larger must go through the multipart upload
flow, which the official CLI and SDKs handle automatically.

## Downloads from cold storage are slow or fail

Objects in the cold storage tier are not instantly retrievable. Downloading
a cold-tier object first requires a "restore" request, which can take up to
4 hours before the object becomes downloadable. Attempting to download
before the restore completes will fail.

## Can I enable object versioning after a bucket already exists?

Versioning must be enabled at bucket creation time. Enabling it retroactively
on an existing bucket is not self-service and requires opening a support
ticket.
