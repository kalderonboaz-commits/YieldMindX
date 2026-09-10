# Deployment specification template

Status: blueprint; hosting and executable deployment configuration remain to be selected. Nothing is provisioned or deployed by extracting this archive.

## Runtime topology

Browser -> HTTPS frontend host -> HTTPS backend -> private database.
A static React build can use static hosting/CDN; server-rendered React needs a server-capable host. Backend runs on a managed app/container platform or equivalent. Prefer managed database operations when suitable for the course and budget. Local Compose is an option for dependencies; Kubernetes is not a prerequisite.

## Environments and configuration

| Environment | Purpose | Data and release |
|---|---|---|
| Local/development | Individual iteration | Synthetic data; isolated local services |
| Staging | Integrated acceptance and migration rehearsal | Separate database and credentials; production-like topology |
| Production | User-facing release | Protected credentials, durable storage and monitored releases |

Use deployment/environments for non-secret settings and references to secret names, never secret values. Each environment has separate access and data. Commit .env.example with names only; local .env files are ignored. Frontend build variables are public. If promoting identical frontend artifacts, inject public runtime configuration at deployment; if environment variables are compiled in, document that separate builds are required.

When choosing containers, place Dockerfiles/build contexts in deployment/containers or document app-local Dockerfiles. Define local services in deployment/local/compose.yaml, pin dependency versions and add health checks and persistent local database volumes. Validate Compose configuration before using it. Production configuration must explicitly account for persistence, TLS, restart behavior and resource limits; do not deploy local defaults unchanged.

## Delivery sequence

1. Pull request: formatting/lint, type checks where applicable, unit/component tests, API/database integration, contract compatibility and builds. Test critical browser flows against isolated services. No production credentials in PR jobs.
2. Merge to protected main: repeat required verification, build release artifacts once where configuration permits, record commit SHA and image digest/artifact checksum, and publish to the selected registry.
3. Deploy staging: apply backward-compatible migrations as a single controlled job, deploy API/frontend, run smoke and critical end-to-end tests; retain evidence.
4. Promote production: enforce the team's environment gate, deploy the verified release, run smoke checks, observe error rate/latency, and record the release and migration versions. Serialize deployments to each environment.
5. On failure: stop promotion, assess migration compatibility, and follow the rollback runbook. Never report deployment success solely because an upload completed.

Select CI hosting during setup. .github is an optional GitHub example; Git alone does not supply CI, branch rules or deployment environments. For GitHub Actions, keep default token permissions read-only, grant narrowly per job, pin third-party actions to reviewed commit SHAs, and use cloud OIDC when supported. Configure environment protections and required checks in repository settings; checked-in files alone do not enforce them.

## Database changes and rollback

Use expand/contract migrations: add compatible schema first, deploy compatible code, backfill safely, and remove old schema in a later release. Do not rewrite migrations already applied in shared environments. Rehearse both fresh installation and upgrades. Run migrations once with the chosen tool's concurrency controls.

Redeploy the prior application artifact only if compatible with the current database. Prefer forward corrective migrations over automatic destructive down-migrations. Backups and point-in-time recovery need configured retention, access controls and a demonstrated restore into an isolated database. Agree acceptable data loss (RPO) and recovery time (RTO); do not invent numeric promises.

## Operational acceptance

Before production, fill in deployment/runbooks with actual commands, service URLs, release owner, health endpoints, rollback decision, migration procedure and restore procedure. Establish structured logs without secrets, request correlation, error/latency metrics, actionable alerts and escalation ownership. Schedule backup checks and practice recovery.

## Sources

Reviewed 2026-09-10: [Docker Compose production guidance](https://docs.docker.com/compose/how-tos/production/), [GitHub deployment environments](https://docs.github.com/en/actions/concepts/workflows-and-actions/deployment-environments), and [GitHub OIDC](https://docs.github.com/en/actions/concepts/security/openid-connect). These support the mechanisms; the release sequence is this scaffold's proposed policy.
