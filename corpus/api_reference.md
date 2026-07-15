# API reference

## Base URL

All API calls are made against `https://api.nimbusstore.example/v1`.

## Authentication

Requests must include a bearer token in the `Authorization` header. Tokens
are issued from the NimbusIAM console and expire 12 hours after issuance;
after expiry, a request will receive a 403 response and a new token must be
generated.

## Core endpoints

- `PUT /buckets/{bucket}/objects/{key}` - upload an object (single request,
  objects up to 100 MB; larger objects must use the multipart upload flow).
- `GET /buckets/{bucket}/objects/{key}` - download an object.
- `DELETE /buckets/{bucket}/objects/{key}` - delete an object.
- `GET /buckets/{bucket}?list` - list objects in a bucket.

## Rate limits

By default, each API token is limited to 200 requests per second on the
Starter and Pro tiers. Enterprise customers can request a higher limit
through their account manager.

## Error codes

- `403` - invalid, missing, or expired bearer token.
- `404` - the bucket or object key does not exist.
- `413` - the object exceeds the 5 TB single-object size limit, or a single
  multipart part exceeds 5 GB.
- `429` - the per-token rate limit was exceeded; the response includes a
  `Retry-After` header indicating how many seconds to wait before retrying.
