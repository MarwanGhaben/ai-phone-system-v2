#!/usr/bin/env bash
# Owner-installed host command. Never mount the Docker socket into Certbot.
set -euo pipefail
umask 077
mode=${1:-renew}
[[ "$mode" == issue || "$mode" == renew ]] || exit 1
email=${2:-}
if [[ "$mode" == issue && ( "$email" != *@* || "$email" == *[[:space:]]* ) ]]; then
    echo ACCOUNT_EMAIL_REQUIRED
    exit 1
fi
[[ $EUID -eq 0 ]] || { echo ROOT_REQUIRED; exit 1; }
cd /opt/ai-phone-system-v2
command -v flock >/dev/null
exec 9>/run/ai-phone-tls.lock
flock -n 9 || { echo TLS_JOB_ALREADY_RUNNING; exit 1; }
domain=aiagent.ghaben.ca
lineage="/opt/ai-phone-system-v2/certbot/conf/live/$domain"
ssl=/opt/ai-phone-system-v2/nginx/ssl
# The existing Compose service has no restart policy; start it if a host reboot
# left it stopped. Certbot's own locks still reject concurrent renewal commands.
docker start ai-voice-certbot >/dev/null

# Confirm certificate and challenge writes target the already mounted host paths.
python3 - <<'PY'
import json, pathlib, subprocess
root = pathlib.Path('/opt/ai-phone-system-v2')
expected = {
 'ai-voice-certbot': {'/etc/letsencrypt': ('certbot/conf', True),
                     '/var/www/certbot': ('nginx/www', True)},
 'ai-voice-nginx': {'/etc/nginx/ssl': ('nginx/ssl', False),
                   '/var/www/certbot': ('nginx/www', False),
                   '/etc/nginx/nginx.conf': ('nginx/nginx.conf', False)}}
for container, paths in expected.items():
    r = subprocess.run(['docker','inspect',container],capture_output=True,check=True)
    mounts = {m['Destination']:m for m in json.loads(r.stdout)[0]['Mounts']}
    for destination, (relative, writable) in paths.items():
        m = mounts.get(destination)
        if not m or pathlib.Path(m['Source']).resolve() != (root/relative).resolve() or m['RW'] != writable:
            raise SystemExit('TLS_MOUNT_MISMATCH_STOPPED')
print('TLS_MOUNTS_VERIFIED')
PY

if [[ "$mode" == issue ]]; then
    backup=$(mktemp -d /opt/ai-phone-tls-backup.XXXXXX)
    cp nginx/nginx.conf "$backup/nginx.conf"
    cp "$ssl/fullchain.pem" "$backup/fullchain.pem"
    cp "$ssl/privkey.pem" "$backup/privkey.pem"
    echo "TLS_BACKUP=$backup"
    # Write in place: nginx's single-file bind mount must retain its inode.
    python3 - <<'PY'
from pathlib import Path
p = Path('nginx/nginx.conf')
s = p.read_text()
route = '''        # Let Certbot prove domain ownership without stopping HTTPS.
        location ^~ /.well-known/acme-challenge/ {
            root /var/www/certbot;
            default_type text/plain;
            try_files $uri =404;
        }

'''
if route not in s:
    marker = '        # Health check only\n'
    if '/.well-known/acme-challenge/' in s or s.count(marker) != 1:
        raise SystemExit('UNEXPECTED_NGINX_CONFIGURATION_STOPPED')
    s = s.replace(marker, route + marker, 1)
    s = s.replace('# HTTP Server (health checks only)', '# HTTP Server (health checks and certificate validation)')
    p.write_text(s)
PY
    if ! docker exec ai-voice-nginx nginx -t; then
        cat "$backup/nginx.conf" > nginx/nginx.conf
        echo NGINX_CONFIG_RESTORED
        exit 1
    fi
    if ! docker exec ai-voice-nginx nginx -s reload; then
        cat "$backup/nginx.conf" > nginx/nginx.conf
        docker exec ai-voice-nginx nginx -s reload || true
        exit 1
    fi
    mkdir -p nginx/www/.well-known/acme-challenge
    chmod 755 nginx/www nginx/www/.well-known nginx/www/.well-known/acme-challenge
    token="ai-phone-check-$(openssl rand -hex 12)"
    probe="nginx/www/.well-known/acme-challenge/$token"
    printf '%s' "$token" > "$probe"
    chmod 644 "$probe"
    trap 'rm -f -- "$probe"' EXIT
    reachable=0
    for attempt in 1 2 3 4 5; do
        if [[ $(curl --noproxy '*' -fsS --connect-timeout 8 --max-time 15 \
            "http://$domain/.well-known/acme-challenge/$token") == "$token" ]]; then
            reachable=1; break
        fi
        sleep 2
    done
    (( reachable )) || { echo PUBLIC_HTTP_CHALLENGE_FAILED; exit 1; }
    rm -f -- "$probe"
    trap - EXIT
    echo PUBLIC_HTTP_CHALLENGE_OK
    docker exec ai-voice-certbot certbot certonly --non-interactive --agree-tos \
        --email "$email" --webroot --webroot-path /var/www/certbot \
        --cert-name "$domain" -d "$domain"
else
    docker exec ai-voice-certbot certbot renew --non-interactive --cert-name "$domain"
fi

openssl x509 -in "$lineage/fullchain.pem" -noout -checkend 604800
openssl x509 -in "$lineage/fullchain.pem" -noout -checkhost "$domain"
cert_key=$(openssl x509 -in "$lineage/fullchain.pem" -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum)
private_key=$(openssl pkey -in "$lineage/privkey.pem" -pubout -outform DER | sha256sum)
[[ "$cert_key" == "$private_key" ]] || { echo CERTIFICATE_KEY_MISMATCH; exit 1; }

if ! cmp -s "$lineage/fullchain.pem" "$ssl/fullchain.pem" || ! cmp -s "$lineage/privkey.pem" "$ssl/privkey.pem"; then
    backup=$(mktemp -d /opt/ai-phone-tls-backup.XXXXXX)
    cp "$ssl/fullchain.pem" "$backup/fullchain.pem"
    cp "$ssl/privkey.pem" "$backup/privkey.pem"
    # Both files are replaced before validation/reload; nginx keeps its current
    # in-memory certificate until a successful reload.
    if ! { install -m 644 "$lineage/fullchain.pem" "$ssl/fullchain.pem" &&
           install -m 600 "$lineage/privkey.pem" "$ssl/privkey.pem" &&
           docker exec ai-voice-nginx nginx -t &&
           docker exec ai-voice-nginx nginx -s reload; }; then
        install -m 644 "$backup/fullchain.pem" "$ssl/fullchain.pem"
        install -m 600 "$backup/privkey.pem" "$ssl/privkey.pem"
        docker exec ai-voice-nginx nginx -s reload || true
        echo TLS_FILES_RESTORED_AFTER_RELOAD_FAILURE
        exit 1
    fi
    echo NGINX_NEW_CERTIFICATE_LOADED
fi
for attempt in 1 2 3 4 5; do
    if curl --noproxy '*' -fsS --connect-timeout 8 --max-time 15 \
        "https://$domain:8443/health" >/dev/null; then
        openssl x509 -in "$ssl/fullchain.pem" -noout -dates
        echo PUBLIC_HTTPS_VALID
        exit 0
    fi
    sleep 2
done
echo PUBLIC_HTTPS_CHECK_FAILED
exit 1
