"""Read-only, allowlisted diagnostics for the owner's failed T005-D rehearsal."""
import json
from pathlib import Path
import re

RELEASE = Path('/opt/ai-phone-persistence-release.5wfu0g_i')
CATEGORIES = {
    'restore_error': r'pg_restore: error:',
    'database_startup_in_progress': r'the database system is starting up',
    'connection_closed': r'server closed the connection unexpectedly|connection.*closed',
    'connection_refused': r'connection refused|connection to server .* failed',
    'name_resolution_failed': r'Name or service not known|Temporary failure in name resolution',
    'schema_contract_rejected': r'SchemaCompatibilityError:',
    'permission_denied': r'permission denied',
    'role_missing': r'ERROR:\s+role .* does not exist',
    'database_missing': r'FATAL:\s+database .* does not exist',
    'relation_missing': r'(?:ERROR:|UndefinedTableError:).*relation .* does not exist',
    'unsupported_setting': r'unrecognized configuration parameter',
    'disk_full': r'no space left on device',
    'memory_failure': r'out of memory|cannot allocate memory|OOMKilled',
    'docker_network_error': r'Error response from daemon:.*network|invalid pool request|address pools',
    'container_not_running': r'container .* is not running',
    'python_import_error': r'ModuleNotFoundError:|ImportError:',
    'migration_rejected': r'^incompatible schema$|^checksum drift$|^unknown version$|^migration failed$|^busy$',
}


def classify(text):
    # Output consists exclusively of fixed labels and line numbers, never source text.
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in CATEGORIES.items():
            if re.search(pattern, line, re.IGNORECASE):
                found.append({'category': label, 'tail_line': number})
    return found[-15:]


def main():
    with (RELEASE / 'private.log').open('rb') as log:
        log.seek(0, 2)
        length = log.tell()
        log.seek(max(0, length - 524288))
        matches = classify(log.read().decode('utf-8', errors='replace'))
    print(json.dumps({
        'status': 'READ_ONLY_DIAGNOSTIC',
        'cutover_started': (RELEASE / 'cutover-started').exists(),
        'deployed_marker': (RELEASE / 'deployed').exists(),
        'categories': matches,
        'unclassified': not matches,
    }, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('{"status":"DIAGNOSTIC_UNAVAILABLE"}')
