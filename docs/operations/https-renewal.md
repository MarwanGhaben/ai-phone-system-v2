# HTTPS recovery and renewal

The observed public endpoint is aiagent.ghaben.ca:8443. Its certificate expired
August 6, 2026; Certbot reported no certificates in its current mounted store.
The account email is supplied explicitly by the owner during installation. HTTP port 80 was
independently reachable from the lead workstation; actual challenge routing is
also checked by the recovery command before issuance.

The owner installs scripts/renew-https.sh as /usr/local/sbin/ai-phone-renew-https
from an exact published commit. Run `ai-phone-renew-https issue <approved-email>` for this recovery.
It validates mounts, backs up the current nginx config/certificate/key in a
root-only /opt/ai-phone-tls-backup.* directory, adds the missing HTTP challenge
location in place, tests/reloads nginx, proves a public challenge response,
and requests the named certificate using Certbot's existing container. Issuance
uses the approved email and accepts Let's Encrypt's subscriber terms. The app,
database and Redis are not stopped or rebuilt. The server nginx file becomes an
intentional tracked modification matching this repository's corrected config;
reconcile it before a later checkout, rather than discarding it.

The command then validates domain, expiry and matching key, installs the certificate
copies nginx serves, tests/reloads nginx, and verifies public HTTPS with normal
certificate validation. A validation/reload failure restores the previous copies.
An issuance failure leaves the working challenge location in place for a corrected
retry; do not force repeated issuance attempts. Private keys are never printed.

Install a host systemd oneshot service calling
`/usr/local/sbin/ai-phone-renew-https renew` and a persistent twice-daily timer.
This host command covers renewal, copying changed files, and reload. The existing
Certbot container's renewal loop may also renew a certificate; the host job installs
those changes on its next successful run. Certbot's own locking rejects overlapping
Certbot operations, and flock serializes host jobs. A transient lock failure is
reported rather than treated as a successful renewal. No Docker socket is exposed
inside the Certbot container.

After first issuance, run `docker exec ai-voice-certbot certbot renew --dry-run
--cert-name aiagent.ghaben.ca`, inspect the timer's next run and verify a real call.
The timer is local server infrastructure, not a Codex reminder. Failures are visible
in `systemctl status ai-phone-tls.service` and its journal; proactive external expiry
monitoring remains separate work. Never delete certbot/conf, nginx/ssl, or recovery
directories as generic cleanup. The existing setup-ssl.sh targets a different domain
and must not be used for this deployment.

References: https://eff-certbot.readthedocs.io/en/stable/using.html#webroot and
https://letsencrypt.org/docs/challenge-types/ .
