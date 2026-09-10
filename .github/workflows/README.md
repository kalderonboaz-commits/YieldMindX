# CI/CD workflow implementation map

This folder is a blueprint, not an active workflow. Add executable YAML only after application manifests, test commands and hosting exist. GitHub is optional; translate the same stages for another Git host.

| Future workflow | Trigger | Jobs and evidence |
|---|---|---|
| ci.yml | Pull request and main push | Locked dependency install; lint/type checks; unit/component tests; isolated database/API integration; contract tests; frontend/backend builds; critical browser tests |
| release.yml | Accepted main revision or release tag | Reuse/produce verified artifacts; record SHA/digests; staging deployment and smoke; protected production promotion and smoke |

Required jobs must fail on test/build failure or missing commands. Publish diagnostics even on failure. Use isolated test data, explicit timeouts, minimum token permissions and reviewed action SHA pins. Cancel superseded PR checks; serialize releases to each environment without interrupting a production deployment midway. PR jobs cannot use production secrets.

Configure required status checks, protected main, environment credentials and approval rules in the Git host. The scaffold does not enable these settings. See [deployment specification](../../project-managment/08_deployment.md) and [test workflow](../../project-managment/09_engineering_workflow.md).
