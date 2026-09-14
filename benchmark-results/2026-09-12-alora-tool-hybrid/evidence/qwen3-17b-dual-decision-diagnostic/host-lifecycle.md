# V19 host lifecycle

The V19 Java diagnostic used one bounded Hetzner Cloud host in `ash`:

- server `165810375`, `modeljars-alora-dual-v19-20260914`;
- firewall `11620744`, `modeljars-alora-dual-v19-ssh`;
- plan `cpx51`, 16 shared vCPU, 32 GB RAM, 360 GB disk;
- live price USD 0.4479/hour;
- created at `2026-09-14T10:42:11Z`;
- hard deletion deadline `2026-09-14T12:41:13Z`; and
- SSH ingress limited to the operator's single `/32` address.

The opening inventory contained zero Hetzner servers and zero firewalls. The provisioning shell
aborted after the create request because two local assignments were accidentally joined. A
read-only inventory found exactly one correctly named and labeled server and firewall, so creation
was not retried and no duplicate resource was produced.

The host checked out scorer revision `487986fcaafd4301124ded60d279fafd42834e76` in detached, clean
state. Uploaded model, adapter, and source-record hashes matched the frozen preflight. Runtime and
benchmark checks passed in 1 minute 35 seconds before the 75-case diagnostic began. Evidence was
copied locally and both files matched their remote SHA-256 values before teardown.

Server deletion action `655265987405934` started at `2026-09-14T11:34:45Z` and completed at
`2026-09-14T11:34:56Z`. Hetzner briefly returned HTTP 422 when firewall deletion was requested while
its attachment state was still reconciling. A read-only check confirmed the server was absent and
the exact labeled firewall was then detached; deleting that same firewall succeeded. Both exact
resource lookups returned HTTP 404 afterward.

The closing provider-wide inventory returned zero servers, firewalls, volumes, floating IPs,
primary IPs, and snapshots. The deadline watchdog was unloaded after that audit. The host existed
for 52 minutes 45 seconds, giving an upper-bound compute cost of approximately USD 0.40.
