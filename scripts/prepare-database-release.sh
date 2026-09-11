#!/usr/bin/env bash
# Prepare an immutable candidate and rehearse only on an isolated database copy.
set -euo pipefail
umask 077
commit=${1:?Pass the reviewed full Git commit}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 1
[[ $EUID -eq 0 ]] || { echo ROOT_REQUIRED; exit 1; }
cd /opt/ai-phone-system-v2
[[ $(git rev-parse "$commit^{commit}") == "$commit" ]]
[[ $(git rev-parse HEAD) == 1affd83320e157343691a44934fac84f7aaa2490 ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
[[ $(docker inspect -f '{{.State.Health.Status}}' ai-voice-app) == healthy ]]
available=$(df -PB1 . | awk 'NR==2 {print $4}')
(( available >= 3221225472 )) || { echo NEED_AT_LEAST_3_GIB_FREE_FOR_BUILD; exit 1; }

release=$(mktemp -d /opt/ai-phone-release.XXXXXX)
echo "RELEASE_DIRECTORY=$release"
old_image=$(docker inspect -f '{{.Image}}' ai-voice-app)
pg_image=$(docker inspect -f '{{.Image}}' ai-voice-db)
rollback="ai-phone-rollback:t005c-${old_image#sha256:}"
candidate="ai-phone-candidate:${commit:0:12}-${release##*.}"
docker tag "$old_image" "$rollback"
printf '%s\n' "$commit" > "$release/commit"
printf '%s\n' "$old_image" > "$release/previous-image-id"
printf '%s\n' "$rollback" > "$release/rollback-tag"
printf '%s\n' "$candidate" > "$release/candidate-tag"
# Protected recovery metadata can contain credentials; never paste these files.
docker inspect ai-voice-app > "$release/previous-app-inspect.json"
cp -p docker-compose.yml "$release/previous-compose.yml"
if [[ -f .env ]]; then cp .env "$release/previous.env"; fi
mkdir "$release/source"
git archive "$commit" | tar -x -C "$release/source"
echo BUILDING_CANDIDATE_CURRENT_APP_STAYS_RUNNING
if ! docker build --label "org.opencontainers.image.revision=$commit" \
    -t "$candidate" "$release/source" > "$release/build.log" 2>&1; then
    echo "BUILD_FAILED_LOG_SAVED=$release/build.log"
    exit 1
fi
docker image inspect -f '{{.Id}}' "$candidate" > "$release/candidate-image-id"
echo CANDIDATE_BUILD_OK
available_mem=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
(( available_mem >= 614400 )) || { echo NEED_600_MIB_AVAILABLE_FOR_REHEARSAL; exit 1; }

docker exec ai-voice-db sh -c \
    'exec pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    > "$release/database.dump" 2> "$release/backup.log"
[[ -s "$release/database.dump" ]]
sha256sum "$release/database.dump" > "$release/database.sha256"
echo FRESH_PROTECTED_BACKUP_CREATED

pg="t005c-restore-${release##*.}"
migrate="t005c-migrate-${release##*.}"
check="t005c-check-${release##*.}"
cleanup() {
    result=$?
    trap - EXIT
    unresolved=0
    for name in "$check" "$migrate" "$pg"; do
        docker rm -fv "$name" >/dev/null 2>&1 || true
        if ! docker container ls -a --format '{{.Names}}' > "$release/cleanup-names.txt"; then
            unresolved=1
        elif grep -Fxq "$name" "$release/cleanup-names.txt"; then
            echo "CLEANUP_REQUIRED=$name"
            unresolved=1
        fi
    done
    if (( unresolved )); then exit 1; fi
    echo REHEARSAL_CONTAINERS_REMOVED
    exit "$result"
}
trap cleanup EXIT
docker run -d --name "$pg" --pull never --network none \
    --memory 192m --memory-swap 192m --cpus 0.5 --pids-limit 64 \
    --tmpfs /var/lib/postgresql/data:rw,noexec,nosuid,size=128m \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -e POSTGRES_USER=t005 -e POSTGRES_PASSWORD=synthetic-rehearsal-only \
    -e POSTGRES_DB=t005 "$pg_image" \
    -c shared_buffers=16MB -c max_connections=12 >/dev/null
ready=0
for ((attempt=0; attempt<60; attempt++)); do
    if docker exec "$pg" pg_isready -h 127.0.0.1 -U t005 -d t005 >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 1
done
(( ready )) || { echo RESTORE_DATABASE_START_FAILED; exit 1; }
if ! docker exec -i "$pg" pg_restore --exit-on-error --single-transaction \
    --no-owner --no-privileges -U t005 -d t005 \
    < "$release/database.dump" > "$release/restore.log" 2>&1; then
    echo "RESTORE_FAILED_LOG_SAVED=$release/restore.log"; exit 1
fi
dsn=postgresql://t005:synthetic-rehearsal-only@127.0.0.1:5432/t005
timeout 90 docker run --rm --name "$migrate" --pull never \
    --network "container:$pg" --memory 192m --memory-swap 192m --pids-limit 64 \
    -e "MIGRATION_DATABASE_URL=$dsn" "$candidate" \
    python -m migrations.runner --prepare
echo RESTORED_DATABASE_MIGRATION_OK
timeout 60 docker run --rm -i --name "$check" --pull never \
    --network "container:$pg" --memory 192m --memory-swap 192m --pids-limit 64 \
    -e "MIGRATION_DATABASE_URL=$dsn" "$candidate" python - <<'PY'
import asyncio
import os
import asyncpg
from migrations.schema_contract import check_runtime_compatibility

async def main():
    conn = await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'], timeout=10,
                                 command_timeout=10)
    try:
        async with conn.transaction(readonly=True):
            await conn.execute("SET LOCAL search_path=pg_catalog")
            await check_runtime_compatibility(conn)
        print('RESTORED_DATABASE_RUNTIME_CONTRACT_OK')
    finally:
        await conn.close(timeout=5)

try:
    asyncio.run(main())
except Exception:
    print('RESTORED_DATABASE_RUNTIME_CONTRACT_FAILED')
    raise SystemExit(1)
PY
[[ $(docker inspect -f '{{.Image}}' ai-voice-app) == "$old_image" ]]
[[ $(docker inspect -f '{{.State.Health.Status}}' ai-voice-app) == healthy ]]
printf '%s\n' "$commit" > "$release/rehearsal-passed"
echo "CANDIDATE_IMAGE_ID=$(cat "$release/candidate-image-id")"
echo "RELEASE_DIRECTORY=$release"
echo RELEASE_PREPARATION_COMPLETE_CURRENT_APP_UNCHANGED
