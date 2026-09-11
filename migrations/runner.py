"""Explicit, bounded T005-A migration entry point; never imported by the app."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

from migrations.schema_contract import (
    BOOTSTRAP_CHECKSUM,
    BOOTSTRAP_VERSION,
    REVISION_CHECKSUM,
    REVISION_VERSION,
    SchemaCompatibilityError,
    TABLE_COLUMNS,
    admin as contract_admin,
    check_application_schema,
    check_runtime_compatibility,
    ledger as contract_ledger,
    relation as contract_relation,
    relation_names,
)

MANIFEST = (("0001", "0001_admin_users_updated_at.sql"),)
# Stable across processes/Python hash randomization; database-local project key.
LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"ai-voice-platform-v2:public:schema_migrations").digest()[:8],
    "big", signed=True,
)
CONNECT_TIMEOUT = 10
COMMAND_TIMEOUT = 15
LOCK_TIMEOUT_MS = 3000
STATEMENT_TIMEOUT_MS = 10000

LEDGER_DDL = """
CREATE TABLE public.schema_migrations (
    version TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE NOT NULL
)
"""


class MigrationError(Exception):
    """Only fixed classifications cross the command-line error boundary."""


def _load_revision():
    _, filename = MANIFEST[0]
    raw = Path(__file__).with_name(filename).read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != REVISION_CHECKSUM:
        raise MigrationError("migration file")
    return raw.decode("utf-8"), checksum


def _load_bootstrap():
    raw = Path(__file__).with_name("bootstrap_schema.sql").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != BOOTSTRAP_CHECKSUM:
        raise MigrationError("migration file")
    return raw.decode("utf-8"), checksum


async def _connect(url):
    # Require explicit destination and identity; never fill gaps from PG* or pgpass.
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in ("postgres", "postgresql") or not parsed.hostname
                or not parsed.username or parsed.password is None
                or not parsed.path.startswith("/") or len(parsed.path) < 2
                or parsed.fragment):
            raise ValueError
        params = dict(host=parsed.hostname, port=parsed.port or 5432,
                      user=unquote(parsed.username), password=unquote(parsed.password),
                      database=unquote(parsed.path[1:]))
    except (ValueError, TypeError):
        raise MigrationError("configuration") from None
    import asyncpg  # Existing dependency; config failure needs no driver import.
    return await asyncpg.connect(
        dsn=url, **params, timeout=CONNECT_TIMEOUT, command_timeout=COMMAND_TIMEOUT,
        server_settings={
            "application_name": "t005-migration-runner",
            "search_path": "pg_catalog",
            "lock_timeout": str(LOCK_TIMEOUT_MS),
            "statement_timeout": str(STATEMENT_TIMEOUT_MS),
            "idle_in_transaction_session_timeout": "15000",
        },
    )


async def _relation(conn, name, *, optional=False):
    try:
        return await contract_relation(conn, name, optional=optional)
    except SchemaCompatibilityError as exc:
        raise MigrationError(str(exc)) from None


async def _admin(conn, *, required_updated=False):
    try:
        return await contract_admin(conn, required_updated=required_updated)
    except SchemaCompatibilityError as exc:
        raise MigrationError(str(exc)) from None


async def _ledger(conn, checksum):
    if checksum != REVISION_CHECKSUM:
        raise MigrationError("migration file")
    try:
        exists, versions = await contract_ledger(conn, allow_bootstrap=True)
    except SchemaCompatibilityError as exc:
        raise MigrationError(str(exc)) from None
    return exists, REVISION_VERSION in versions


async def _record_revision(conn, version, checksum):
    await conn.execute(
        "INSERT INTO public.schema_migrations (version,checksum,applied_at) "
        "VALUES ($1,$2,CURRENT_TIMESTAMP)", version, checksum,
    )


async def _execute(conn, *, apply, sql, checksum):
    # Read-only status never takes advisory locks or issues DDL/DML/sequence calls.
    if apply:
        if not await conn.fetchval(
            "SELECT pg_catalog.pg_try_advisory_xact_lock($1::bigint)", LOCK_KEY,
        ):
            raise MigrationError("busy")
        await _relation(conn, "admin_users")
        # Hold against non-runner DDL/writes until commit; bounded by lock_timeout.
        await conn.execute("LOCK TABLE public.admin_users IN ACCESS EXCLUSIVE MODE")
        if await _relation(conn, "schema_migrations", optional=True) is not None:
            await conn.execute("LOCK TABLE public.schema_migrations IN ACCESS EXCLUSIVE MODE")
    has_updated = await _admin(conn)
    ledger_exists, recorded = await _ledger(conn, checksum)
    if recorded:
        await _admin(conn, required_updated=True)
        return "up-to-date 0001"
    if not apply:
        return "pending 0001"
    # Entire predecessor validation above precedes the first schema mutation.
    if not ledger_exists:
        await conn.execute(LEDGER_DDL)
    if not has_updated:
        await conn.execute(sql)
    await _admin(conn, required_updated=True)
    await _record_revision(conn, MANIFEST[0][0], checksum)
    await _ledger(conn, checksum)
    return "applied 0001"


async def _public_is_empty(conn):
    return not await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','S','f')
        ) OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_proc p
            JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname='public'
        ) OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_type t
            JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace
            WHERE n.nspname='public'
        )
    """)


async def _prepare_execute(conn, *, revision_sql, revision_checksum,
                           bootstrap_sql, bootstrap_checksum):
    if not await conn.fetchval(
        "SELECT pg_catalog.pg_try_advisory_xact_lock($1::bigint)", LOCK_KEY,
    ):
        raise MigrationError("busy")

    if await _public_is_empty(conn):
        await conn.execute(bootstrap_sql)
        try:
            await check_application_schema(conn)
        except SchemaCompatibilityError as exc:
            raise MigrationError(str(exc)) from None
        await _admin(conn, required_updated=True)
        await conn.execute(LEDGER_DDL)
        try:
            await contract_ledger(conn, allow_bootstrap=True)
        except SchemaCompatibilityError as exc:
            raise MigrationError(str(exc)) from None
        await _record_revision(conn, BOOTSTRAP_VERSION, bootstrap_checksum)
        await _record_revision(conn, REVISION_VERSION, revision_checksum)
        try:
            await check_runtime_compatibility(conn)
        except SchemaCompatibilityError as exc:
            raise MigrationError(str(exc)) from None
        return "prepared bootstrap-v1 0001"

    try:
        names = await relation_names(conn)
        allowed = set(TABLE_COLUMNS) | {"schema_migrations"}
        if not set(TABLE_COLUMNS).issubset(names) or not names.issubset(allowed):
            raise SchemaCompatibilityError("incompatible schema")
        lock_targets = [
            'public."' + name.replace('"', '""') + '"' for name in sorted(TABLE_COLUMNS)
        ]
        if "schema_migrations" in names:
            lock_targets.append("public.schema_migrations")
        await conn.execute("LOCK TABLE " + ",".join(lock_targets) + " IN SHARE MODE")
        await check_application_schema(
            conn, allow_missing_admin_updated=True, ledger_optional=True)
        _, versions = await contract_ledger(conn, allow_bootstrap=True)
    except SchemaCompatibilityError as exc:
        raise MigrationError(str(exc)) from None

    if versions:
        await _admin(conn, required_updated=True)
        try:
            await check_runtime_compatibility(conn)
        except SchemaCompatibilityError as exc:
            raise MigrationError(str(exc)) from None
        return "up-to-date 0001"

    result = await _execute(
        conn, apply=True, sql=revision_sql, checksum=revision_checksum)
    try:
        await check_runtime_compatibility(conn)
    except SchemaCompatibilityError as exc:
        raise MigrationError(str(exc)) from None
    return result


async def run(*, apply=False, prepare=False):
    if apply and prepare:
        raise MigrationError(
            "arguments: choose exactly one of --status / --apply / --prepare")
    url = os.environ.get("MIGRATION_DATABASE_URL")
    if not url or not url.strip():
        raise MigrationError("configuration")
    try:
        sql, checksum = _load_revision()
        bootstrap_sql = bootstrap_checksum = None
        if prepare:
            bootstrap_sql, bootstrap_checksum = _load_bootstrap()
    except Exception:
        raise MigrationError("migration file") from None
    try:
        conn = await _connect(url)
    except MigrationError:
        raise
    except Exception:
        raise MigrationError("connection") from None
    try:
        async with conn.transaction(
            isolation="read_committed" if (apply or prepare) else "repeatable_read",
            readonly=not (apply or prepare),
        ):
            if prepare:
                return await _prepare_execute(
                    conn, revision_sql=sql, revision_checksum=checksum,
                    bootstrap_sql=bootstrap_sql, bootstrap_checksum=bootstrap_checksum)
            return await _execute(conn, apply=apply, sql=sql, checksum=checksum)
    except MigrationError:
        raise
    except Exception as exc:
        category = "busy" if getattr(exc, "sqlstate", None) == "55P03" else "migration failed"
        raise MigrationError(category) from None
    finally:
        try:
            await conn.close(timeout=5)
        except BaseException:
            # asyncpg terminate is synchronous and releases this session's lock
            # even when cancellation or a failed close interrupts normal cleanup.
            conn.terminate()


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes unknown arguments, which could contain a DSN.
        raise MigrationError(
            "arguments: choose exactly one of --status / --apply / --prepare")


def main(argv=None):
    parser = _Parser(prog="python -m migrations.runner", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    try:
        args = parser.parse_args(argv)
        result = asyncio.run(run(apply=args.apply, prepare=args.prepare))
        print(result)
        return 0
    except MigrationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1
    except Exception:
        print("migration failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
