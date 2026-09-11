#!/usr/bin/env bash
set -euo pipefail
umask 077
commit=${1:?Pass the reviewed script commit}
email=${2:?Pass the approved account email}
[[ "$commit" =~ ^[0-9a-f]{40}$ && $EUID -eq 0 ]] || exit 1
cd /opt/ai-phone-system-v2
command -v python3 >/dev/null
command -v systemctl >/dev/null
command -v openssl >/dev/null
command -v flock >/dev/null
systemctl is-active --quiet docker || { echo DOCKER_SERVICE_NOT_ACTIVE; exit 1; }
backup=$(mktemp -d /opt/ai-phone-tls-setup.XXXXXX)
for file in /usr/local/sbin/ai-phone-renew-https /etc/systemd/system/ai-phone-tls.service /etc/systemd/system/ai-phone-tls.timer; do
    if [[ -f "$file" ]]; then cp "$file" "$backup/$(basename "$file")"; fi
done
git show "$commit:scripts/renew-https.sh" > "$backup/renew-https.sh"
bash -n "$backup/renew-https.sh"
install -m 700 "$backup/renew-https.sh" /usr/local/sbin/ai-phone-renew-https
/usr/local/sbin/ai-phone-renew-https issue "$email"

cat > /etc/systemd/system/ai-phone-tls.service <<'SERVICE'
[Unit]
Description=Renew and deploy the AI phone HTTPS certificate
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/ai-phone-renew-https renew
TimeoutStartSec=600
SERVICE
cat > /etc/systemd/system/ai-phone-tls.timer <<'TIMER'
[Unit]
Description=Twice-daily AI phone certificate renewal and deployment

[Timer]
OnCalendar=*-*-* 03,15:00:00
RandomizedDelaySec=1800
Persistent=true

[Install]
WantedBy=timers.target
TIMER
chmod 644 /etc/systemd/system/ai-phone-tls.service /etc/systemd/system/ai-phone-tls.timer
systemctl daemon-reload
systemctl enable --now ai-phone-tls.timer
systemctl is-active --quiet ai-phone-tls.timer
echo AUTOMATIC_TLS_TIMER_ENABLED
if ! docker exec ai-voice-certbot certbot renew --non-interactive --dry-run --cert-name aiagent.ghaben.ca; then
    echo HTTPS_REPAIRED_BUT_RENEWAL_DRY_RUN_NEEDS_REVIEW
    exit 1
fi
systemctl list-timers ai-phone-tls.timer --no-pager
echo HTTPS_REPAIR_AND_RENEWAL_TEST_COMPLETE
