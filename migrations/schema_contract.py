"""Read-only structural compatibility contract shared by startup and migrations."""
from __future__ import annotations
import json

BOOTSTRAP_VERSION = "bootstrap-v1"
BOOTSTRAP_CHECKSUM = "1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8"
REVISION_VERSION = "0001"
REVISION_CHECKSUM = "53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9"
BOOKINGS_REVISION_VERSION = "0002"
BOOKINGS_REVISION_CHECKSUM = "62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6"
OBSERVATION_REVISION_VERSION = "0003"
OBSERVATION_REVISION_CHECKSUM = "b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830"
OBSERVATION_TABLE = "booking_provider_observations"
OBSERVATION_CONTROL_TABLE = "booking_provider_observation_control"


class SchemaCompatibilityError(Exception):
    """A fixed, safe compatibility classification."""


# name:type[:varchar length][!] where ! means NOT NULL. This is an explicit,
# reviewed contract rather than an expectation derived from the SQL under test.
_SPECS = {
    "admin_sessions": "id:int4! user_id:int4 session_token:varchar:255! created_at:timestamp expires_at:timestamp! ip_address:varchar:45 user_agent:text",
    "admin_users": "id:int4! username:varchar:100! email:varchar:255! password_hash:varchar:255! is_active:bool is_superuser:bool created_at:timestamp updated_at:timestamp last_login:timestamp",
    "analytics_events": "id:int4! tenant_id:int4 call_sid:varchar:100 event_type:varchar:100! event_data:jsonb duration_ms:int4 timestamp:timestamptz",
    "api_usage": "id:int4! service_name:varchar:100! request_count:int4 tokens_used:int4 characters_used:int4 audio_seconds:float8 estimated_cost:numeric recorded_at:date",
    "appointments": "id:int4! call_sid:varchar:100 tenant_id:int4 client_name:varchar:100! client_phone:varchar:50! client_email:varchar:100 accountant_name:varchar:100 appointment_time:timestamptz! appointment_time_formatted:varchar:200 client_type:varchar:50 language:varchar:10 status:varchar:50 booking_url:text event_type_uri:varchar:255 service_id:varchar:100 staff_id:varchar:100 ms_booking_id:varchar:100 reminder_sent:bool reminder_sent_at:timestamptz created_at:timestamptz updated_at:timestamptz",
    "bookings": "id:int4! call_sid:varchar:100 phone_number:varchar:50 client_name:varchar:255 client_email:varchar:255 accountant_name:varchar:255 appointment_time:timestamp client_type:varchar:50 language:varchar:10 status:varchar:50 created_at:timestamp ms_booking_id:varchar:255 notes:text appointment_time_utc:timestamptz",
    "call_logs": "id:int4! call_sid:varchar:100 phone_number:varchar:50 caller_name:varchar:255 language:varchar:10 started_at:timestamp ended_at:timestamp duration_seconds:int4 status:varchar:50 transfer_requested:bool dtmf_count:int4 booking_made:bool notes:text",
    "callers": "id:int4! phone_number:varchar:50! name:varchar:100! language:varchar:10 call_count:int4 first_call:timestamptz last_call:timestamptz tenant_id:int4 created_at:timestamptz updated_at:timestamptz",
    "calls": "id:int4! call_sid:varchar:100! tenant_id:int4 phone_number:varchar:50 caller_name:varchar:100 language:varchar:10 detected_language:varchar:10 direction:varchar:20 status:varchar:50 started_at:timestamptz ended_at:timestamptz duration_seconds:int4 conversation_turns:int4 transferred:bool transferred_to:varchar:100 recording_url:text transcription:text error_message:text created_at:timestamptz updated_at:timestamptz",
    "conversation_turns": "id:int4! call_sid:varchar:100 role:varchar:20! content:text! language:varchar:10 audio_duration_ms:int4 stt_confidence:float8 intent:varchar:100 entities:jsonb timestamp:timestamptz",
    "conversations": "id:int4! call_sid:varchar:100 tenant_id:int4 state:jsonb intent_history:jsonb summary:text sentiment:varchar:50 created_at:timestamptz updated_at:timestamptz",
    "knowledge_articles": "id:int4! tenant_id:int4 category:varchar:100 question:text! answer:text! language:varchar:10 priority:int4 tags:_text is_active:bool created_at:timestamptz updated_at:timestamptz",
    "mfa_codes": "id:int4! user_id:int4 code:varchar:6! created_at:timestamp expires_at:timestamp! used:bool",
    "sms_logs": "id:int4! phone_number:varchar:50 client_name:varchar:255 message:text provider:varchar:50 status:varchar:50 sent_at:timestamp booking_id:int4 booking_link:varchar:500 error_message:text",
    "system_metrics": "id:int4! cpu_percent:float8 memory_percent:float8 disk_percent:float8 active_calls:int4 recorded_at:timestamp",
    "tenants": "id:int4! name:varchar:100! slug:varchar:50! industry:varchar:100 phone_number:varchar:50 is_active:bool settings:jsonb created_at:timestamptz updated_at:timestamptz",
    "users": "id:int4! tenant_id:int4 username:varchar:50! password_hash:varchar:255! email:varchar:100 role:varchar:50 mfa_enabled:bool mfa_secret:varchar:100 last_login:timestamptz is_active:bool created_at:timestamptz updated_at:timestamptz",
}


def _parse_spec(spec):
    result = {}
    for token in spec.split():
        required = token.endswith("!")
        bits = token.rstrip("!").split(":")
        name, type_name = bits[:2]
        if type_name == "varchar":
            typmod = int(bits[2]) + 4
        elif type_name == "numeric":
            typmod = 655368  # DECIMAL(10,4)
        else:
            typmod = -1
        result[name] = (type_name, typmod, required)
    return result


TABLE_COLUMNS = {name: _parse_spec(spec) for name, spec in _SPECS.items()}
UNIQUE_KEYS = {
    ("admin_sessions", ("session_token",)), ("admin_users", ("email",)),
    ("admin_users", ("username",)), ("api_usage", ("service_name", "recorded_at")),
    ("call_logs", ("call_sid",)), ("callers", ("phone_number",)),
    ("calls", ("call_sid",)), ("tenants", ("slug",)), ("users", ("username",)),
}
FOREIGN_KEYS = {
    ("admin_sessions", ("user_id",), "admin_users", ("id",), "c"),
    ("appointments", ("call_sid",), "calls", ("call_sid",), "n"),
    ("conversation_turns", ("call_sid",), "calls", ("call_sid",), "c"),
    ("conversations", ("call_sid",), "calls", ("call_sid",), "c"),
    ("mfa_codes", ("user_id",), "admin_users", ("id",), "c"),
    ("sms_logs", ("booking_id",), "bookings", ("id",), "a"),
}
ADMIN_COLUMNS = {key: value for key, value in TABLE_COLUMNS["admin_users"].items()
                 if key != "updated_at"}
LEDGER_COLUMNS = {
    "version": ("text", -1, True),
    "checksum": ("text", -1, True),
    "applied_at": ("timestamptz", -1, True),
}
OBSERVATION_COLUMNS = _parse_spec(
    "booking_id:int4! snapshot_provider_id:varchar:255! snapshot_start:timestamptz! "
    "snapshot_status:varchar:50! checked_at:timestamptz! outcome:varchar:20! "
    "error_category:varchar:32 http_status:int4 provider_start:timestamptz "
    "provider_end:timestamptz staff_member_ids:_text service_id:text is_location_online:bool "
    "observed_provider_id:varchar:255"
)
OBSERVATION_CHECKS = frozenset({
    "booking_observation_snapshot_status_check",
    "booking_observation_outcome_check",
    "booking_observation_error_check",
    "booking_observation_error_category_check",
    "booking_observation_interval_pair_check",
    "booking_observation_interval_order_check",
    "booking_observation_http_status_check",
    "booking_observation_identity_presence_check",
    "booking_observation_identity_match_check",
})
OBSERVATION_CONTROL_CHECK = "booking_observation_control_singleton_check"
OBSERVATION_CONTROL_COLUMNS = _parse_spec(
    "singleton:int4! next_request_at:timestamptz! check_definitions:jsonb!"
)


async def check_observation_schema(conn):
    """Require the exact table shape, protective constraints and bounded index."""
    oid = await relation(conn, OBSERVATION_TABLE)
    actual = await columns(conn, oid)
    check_columns(actual, OBSERVATION_COLUMNS)
    if any(row["default_expr"] is not None for row in actual.values()):
        raise SchemaCompatibilityError("incompatible schema")
    rows = await conn.fetch("""
        SELECT k.conname,k.contype::text,k.condeferrable,k.condeferred,
               k.convalidated,k.connoinherit,
               pg_catalog.pg_get_constraintdef(k.oid, false) AS definition,
               rn.relname AS referenced_table,ns.nspname AS referenced_schema,
               k.confdeltype::text,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.conkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=k.conrelid AND a.attnum=u.num ORDER BY u.ord) AS columns,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.confkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=k.confrelid AND a.attnum=u.num ORDER BY u.ord) AS ref_columns
        FROM pg_catalog.pg_constraint k
        LEFT JOIN pg_catalog.pg_class rn ON rn.oid=k.confrelid
        LEFT JOIN pg_catalog.pg_namespace ns ON ns.oid=rn.relnamespace
        WHERE k.conrelid=$1
    """, oid)
    expected_names = {
        OBSERVATION_TABLE + "_pkey", OBSERVATION_TABLE + "_booking_id_fkey",
        *OBSERVATION_CHECKS,
    }
    if {row["conname"] for row in rows} != expected_names or len(rows) != len(expected_names):
        raise SchemaCompatibilityError("incompatible schema")
    definitions = {}
    for row in rows:
        if (row["condeferrable"] or row["condeferred"]
                or not row["convalidated"]):
            raise SchemaCompatibilityError("incompatible schema")
        name = row["conname"]
        if name.endswith("_pkey"):
            if row["contype"] != "p" or tuple(row["columns"]) != ("booking_id",):
                raise SchemaCompatibilityError("incompatible schema")
        elif name.endswith("_booking_id_fkey"):
            if (row["contype"] != "f" or tuple(row["columns"]) != ("booking_id",)
                    or row["referenced_schema"] != "public"
                    or row["referenced_table"] != "bookings"
                    or tuple(row["ref_columns"]) != ("id",)
                    or row["confdeltype"] != "c"):
                raise SchemaCompatibilityError("incompatible schema")
        else:
            if row["contype"] != "c" or row["connoinherit"]:
                raise SchemaCompatibilityError("incompatible schema")
            definitions[name] = row["definition"]
    indexes = await conn.fetch("""
        SELECT c.relname,i.indisvalid,i.indisready,i.indisunique,
               i.indpred IS NOT NULL AS partial,i.indexprs IS NOT NULL AS expression,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(i.indkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=i.indrelid AND a.attnum=u.num ORDER BY u.ord) AS columns
        FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid
        WHERE i.indrelid=$1
    """, oid)
    by_name = {row["relname"]: row for row in indexes}
    if set(by_name) != {OBSERVATION_TABLE + "_pkey", OBSERVATION_TABLE + "_checked_at_idx"}:
        raise SchemaCompatibilityError("incompatible schema")
    if (tuple(by_name[OBSERVATION_TABLE + "_pkey"]["columns"]) != ("booking_id",)
            or not by_name[OBSERVATION_TABLE + "_pkey"]["indisunique"]
            or tuple(by_name[OBSERVATION_TABLE + "_checked_at_idx"]["columns"])
            != ("checked_at", "booking_id")
            or by_name[OBSERVATION_TABLE + "_checked_at_idx"]["indisunique"]
            or any(not row["indisvalid"] or not row["indisready"]
                   or row["partial"] or row["expression"] for row in indexes)):
        raise SchemaCompatibilityError("incompatible schema")

    # PostgreSQL's own deparser captured these exact predicates when the
    # checksum-pinned 0003 asset was applied. Compare the current definitions
    # against that recorded migration-time baseline, not token vocabulary.
    control_oid = await relation(conn, OBSERVATION_CONTROL_TABLE)
    control_columns = await columns(conn, control_oid)
    check_columns(control_columns, OBSERVATION_CONTROL_COLUMNS)
    if any(row["default_expr"] is not None for row in control_columns.values()):
        raise SchemaCompatibilityError("incompatible schema")
    control_constraints = await conn.fetch("""
        SELECT k.conname,k.contype::text,k.condeferrable,k.condeferred,
               k.convalidated,k.connoinherit,
               pg_catalog.pg_get_constraintdef(k.oid, false) AS definition,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.conkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=k.conrelid AND a.attnum=u.num ORDER BY u.ord) AS columns
        FROM pg_catalog.pg_constraint k WHERE k.conrelid=$1
    """, control_oid)
    if ({row["conname"] for row in control_constraints}
            != {OBSERVATION_CONTROL_TABLE + "_pkey", OBSERVATION_CONTROL_CHECK}
            or len(control_constraints) != 2):
        raise SchemaCompatibilityError("incompatible schema")
    for row in control_constraints:
        if (row["condeferrable"] or row["condeferred"]
                or not row["convalidated"] or tuple(row["columns"]) != ("singleton",)):
            raise SchemaCompatibilityError("incompatible schema")
        if row["conname"] == OBSERVATION_CONTROL_CHECK:
            if row["contype"] != "c" or row["connoinherit"]:
                raise SchemaCompatibilityError("incompatible schema")
            definitions[row["conname"]] = row["definition"]
        elif row["contype"] != "p":
            raise SchemaCompatibilityError("incompatible schema")
    control_indexes = await conn.fetch("""
        SELECT c.relname,i.indisvalid,i.indisready,i.indisunique,i.indimmediate,
               i.indpred IS NOT NULL AS partial,i.indexprs IS NOT NULL AS expression,
               i.indnatts,i.indnkeyatts,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(i.indkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=i.indrelid AND a.attnum=u.num ORDER BY u.ord) AS columns
        FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid
        WHERE i.indrelid=$1
    """, control_oid)
    if (len(control_indexes) != 1
            or control_indexes[0]["relname"] != OBSERVATION_CONTROL_TABLE + "_pkey"
            or tuple(control_indexes[0]["columns"]) != ("singleton",)
            or not all(control_indexes[0][key] for key in
                       ("indisvalid", "indisready", "indisunique", "indimmediate"))
            or control_indexes[0]["partial"] or control_indexes[0]["expression"]
            or control_indexes[0]["indnatts"] != 1
            or control_indexes[0]["indnkeyatts"] != 1):
        raise SchemaCompatibilityError("incompatible schema")
    control_rows = await conn.fetch(
        "SELECT singleton,check_definitions FROM public.booking_provider_observation_control")
    if len(control_rows) != 1 or control_rows[0]["singleton"] != 1:
        raise SchemaCompatibilityError("incompatible schema")
    try:
        baseline = control_rows[0]["check_definitions"]
        if isinstance(baseline, str):
            baseline = json.loads(baseline)
    except (TypeError, ValueError):
        raise SchemaCompatibilityError("incompatible schema") from None
    if baseline != definitions:
        raise SchemaCompatibilityError("incompatible schema")


async def relation(conn, name, *, optional=False, allow_triggers=False):
    row = await conn.fetchrow("""
        SELECT c.oid,c.relkind::text,c.relpersistence::text,c.relispartition,
               c.relrowsecurity,c.relforcerowsecurity,c.relhasrules,
               EXISTS (SELECT 1 FROM pg_catalog.pg_inherits i
                       WHERE i.inhrelid=c.oid OR i.inhparent=c.oid) AS inherited,
               EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
                       WHERE t.tgrelid=c.oid AND NOT t.tgisinternal) AS user_triggers
        FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n
          ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relname=$1
    """, name)
    if row is None:
        if optional:
            return None
        raise SchemaCompatibilityError("incompatible schema")
    flags = ("relispartition", "relrowsecurity", "relforcerowsecurity",
             "relhasrules", "inherited")
    if (row["relkind"] != "r" or row["relpersistence"] != "p"
            or any(row[key] for key in flags)
            or (row["user_triggers"] and not allow_triggers)):
        raise SchemaCompatibilityError("incompatible schema")
    return row["oid"]


async def columns(conn, oid):
    rows = await conn.fetch("""
        SELECT a.attname,t.typname,n.nspname,a.atttypmod,a.attnotnull,
               a.attidentity::text,a.attgenerated::text,a.attisdropped,
               pg_catalog.pg_get_expr(d.adbin,d.adrelid) AS default_expr
        FROM pg_catalog.pg_attribute a
        LEFT JOIN pg_catalog.pg_type t ON t.oid=a.atttypid
        LEFT JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace
        LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
        WHERE a.attrelid=$1 AND a.attnum>0 ORDER BY a.attnum
    """, oid)
    result = {}
    for row in rows:
        if (row["attisdropped"] or row["attidentity"] or row["attgenerated"]
                or row["nspname"] != "pg_catalog" or row["attname"] in result):
            raise SchemaCompatibilityError("incompatible schema")
        result[row["attname"]] = row
    return result


def check_columns(actual, expected):
    if set(actual) != set(expected):
        raise SchemaCompatibilityError("incompatible schema")
    for name, shape in expected.items():
        row = actual[name]
        if (row["typname"], row["atttypmod"], row["attnotnull"]) != shape:
            raise SchemaCompatibilityError("incompatible schema")


async def keys(conn, oid, expected, *, ledger_table=False):
    rows = await conn.fetch("""
        SELECT k.contype::text,k.condeferrable,k.condeferred,k.convalidated,
               i.indisvalid,i.indisready,i.indisunique,i.indimmediate,
               i.indpred IS NOT NULL AS partial,i.indexprs IS NOT NULL AS expression,
               i.indnatts,i.indnkeyatts,k.conindid,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.conkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=k.conrelid AND a.attnum=u.num ORDER BY u.ord) AS columns
        FROM pg_catalog.pg_constraint k LEFT JOIN pg_catalog.pg_index i
          ON i.indexrelid=k.conindid WHERE k.conrelid=$1
    """, oid)
    found = set()
    for row in rows:
        key = (row["contype"], tuple(row["columns"]))
        index_ok = row["contype"] == "f" or (
            all(row[item] for item in ("indisvalid", "indisready", "indisunique", "indimmediate"))
            and not row["partial"] and not row["expression"]
            and row["indnatts"] == len(key[1]) and row["indnkeyatts"] == len(key[1]))
        if (key not in expected or key in found or row["condeferrable"]
                or row["condeferred"] or not row["convalidated"] or not index_ok):
            raise SchemaCompatibilityError("incompatible schema")
        found.add(key)
    if found != expected:
        raise SchemaCompatibilityError("incompatible schema")
    if ledger_table:
        indexes = await conn.fetch(
            "SELECT indexrelid FROM pg_catalog.pg_index WHERE indrelid=$1", oid)
        if {row["indexrelid"] for row in indexes} != {row["conindid"] for row in rows}:
            raise SchemaCompatibilityError("incompatible schema")


async def admin(conn, *, required_updated=False):
    oid = await relation(conn, "admin_users")
    actual = await columns(conn, oid)
    expected = dict(ADMIN_COLUMNS)
    if "updated_at" in actual:
        expected["updated_at"] = ("timestamp", -1, False)
        if actual["updated_at"]["default_expr"] not in (None, "now()", "CURRENT_TIMESTAMP"):
            raise SchemaCompatibilityError("incompatible schema")
    elif required_updated:
        raise SchemaCompatibilityError("incompatible schema")
    check_columns(actual, expected)
    await keys(conn, oid, {("p", ("id",)), ("u", ("username",)), ("u", ("email",))})
    return "updated_at" in actual


async def ledger(conn, *, allow_bootstrap=False, require_revision=False,
                 require_current=False, require_observation=False):
    oid = await relation(conn, "schema_migrations", optional=True)
    if oid is None:
        if require_revision or require_current or require_observation:
            raise SchemaCompatibilityError("incompatible schema")
        return False, frozenset()
    actual = await columns(conn, oid)
    check_columns(actual, LEDGER_COLUMNS)
    if any(row["default_expr"] is not None for row in actual.values()):
        raise SchemaCompatibilityError("incompatible schema")
    await keys(conn, oid, {("p", ("version",))}, ledger_table=True)
    rows = await conn.fetch(
        "SELECT version,checksum,pg_catalog.isfinite(applied_at) AS finite,applied_at "
        "FROM public.schema_migrations")
    expected = {
        REVISION_VERSION: REVISION_CHECKSUM,
        BOOKINGS_REVISION_VERSION: BOOKINGS_REVISION_CHECKSUM,
        OBSERVATION_REVISION_VERSION: OBSERVATION_REVISION_CHECKSUM,
    }
    if allow_bootstrap:
        expected[BOOTSTRAP_VERSION] = BOOTSTRAP_CHECKSUM
    versions, applied = set(), {}
    for row in rows:
        version = row["version"]
        if version not in expected:
            raise SchemaCompatibilityError("unknown version")
        if version in versions or not row["finite"]:
            raise SchemaCompatibilityError("incompatible schema")
        if row["checksum"] != expected[version]:
            raise SchemaCompatibilityError("checksum drift")
        versions.add(version)
        applied[version] = row["applied_at"]
    if BOOTSTRAP_VERSION in versions and REVISION_VERSION not in versions:
        raise SchemaCompatibilityError("incompatible schema")
    if BOOKINGS_REVISION_VERSION in versions and REVISION_VERSION not in versions:
        raise SchemaCompatibilityError("incompatible schema")
    if (OBSERVATION_REVISION_VERSION in versions
            and BOOKINGS_REVISION_VERSION not in versions):
        raise SchemaCompatibilityError("incompatible schema")
    if (BOOTSTRAP_VERSION in versions
            and applied[BOOTSTRAP_VERSION] > applied[REVISION_VERSION]):
        raise SchemaCompatibilityError("incompatible schema")
    if require_revision and REVISION_VERSION not in versions:
        raise SchemaCompatibilityError("incompatible schema")
    if (REVISION_VERSION in versions and BOOKINGS_REVISION_VERSION in versions
            and applied[REVISION_VERSION] > applied[BOOKINGS_REVISION_VERSION]):
        raise SchemaCompatibilityError("incompatible schema")
    if require_current and BOOKINGS_REVISION_VERSION not in versions:
        raise SchemaCompatibilityError("incompatible schema")
    if (BOOKINGS_REVISION_VERSION in versions and OBSERVATION_REVISION_VERSION in versions
            and applied[BOOKINGS_REVISION_VERSION] > applied[OBSERVATION_REVISION_VERSION]):
        raise SchemaCompatibilityError("incompatible schema")
    if require_observation and OBSERVATION_REVISION_VERSION not in versions:
        raise SchemaCompatibilityError("incompatible schema")
    return True, frozenset(versions)


async def relation_names(conn):
    rows = await conn.fetch("""
        SELECT c.relname FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','f')
    """)
    return {row["relname"] for row in rows}


async def check_application_schema(conn, *, allow_missing_admin_updated=False,
                                   allow_missing_booking_aware=False,
                                   allow_missing_observations=False,
                                   ledger_optional=True):
    names = await relation_names(conn)
    expected_names = set(TABLE_COLUMNS)
    if "schema_migrations" in names:
        expected_names.add("schema_migrations")
    elif not ledger_optional:
        raise SchemaCompatibilityError("incompatible schema")
    if OBSERVATION_TABLE in names:
        expected_names.update((OBSERVATION_TABLE, OBSERVATION_CONTROL_TABLE))
    elif not allow_missing_observations:
        raise SchemaCompatibilityError("incompatible schema")
    if names != expected_names:
        raise SchemaCompatibilityError("incompatible schema")
    lock_names = sorted(TABLE_COLUMNS)
    if "schema_migrations" in names:
        lock_names.append("schema_migrations")
    if OBSERVATION_TABLE in names:
        lock_names.extend((OBSERVATION_TABLE, OBSERVATION_CONTROL_TABLE))
    lock_targets = [
        'public."' + name.replace('"', '""') + '"' for name in lock_names
    ]
    await conn.execute(
        "LOCK TABLE " + ",".join(lock_targets) + " IN ACCESS SHARE MODE")

    expected_constraints = {
        (table, "p", ("id",), None, (), None) for table in TABLE_COLUMNS
    }
    expected_constraints |= {
        (table, "u", cols, None, (), None) for table, cols in UNIQUE_KEYS
    }
    expected_constraints |= {
        (table, "f", cols, target, target_cols, delete)
        for table, cols, target, target_cols, delete in FOREIGN_KEYS
    }
    actual_constraints = set()
    for table, expected in TABLE_COLUMNS.items():
        oid = await relation(conn, table, allow_triggers=True)
        actual = await columns(conn, oid)
        if table == "admin_users" and allow_missing_admin_updated and "updated_at" not in actual:
            expected = {name: shape for name, shape in expected.items() if name != "updated_at"}
        if (table == "bookings" and allow_missing_booking_aware
                and "appointment_time_utc" not in actual):
            expected = {
                name: shape for name, shape in expected.items()
                if name != "appointment_time_utc"
            }
        check_columns(actual, expected)
        if (table == "bookings" and "appointment_time_utc" in actual
                and actual["appointment_time_utc"]["default_expr"] is not None):
            raise SchemaCompatibilityError("incompatible schema")
        rows = await conn.fetch("""
            SELECT k.contype::text,k.condeferrable,k.condeferred,k.convalidated,
                   rn.relname AS referenced_table,ns.nspname AS referenced_schema,
                   k.confdeltype::text,
                   ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.conkey)
                         WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                         ON a.attrelid=k.conrelid AND a.attnum=u.num ORDER BY u.ord) AS columns,
                   ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.confkey)
                         WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                         ON a.attrelid=k.confrelid AND a.attnum=u.num ORDER BY u.ord) AS ref_columns,
                   i.indisvalid,i.indisready,i.indisunique,i.indimmediate,
                   i.indpred IS NOT NULL AS partial,i.indexprs IS NOT NULL AS expression,
                   i.indnatts,i.indnkeyatts
            FROM pg_catalog.pg_constraint k
            LEFT JOIN pg_catalog.pg_class rn ON rn.oid=k.confrelid
            LEFT JOIN pg_catalog.pg_namespace ns ON ns.oid=rn.relnamespace
            LEFT JOIN pg_catalog.pg_index i ON i.indexrelid=k.conindid
            WHERE k.conrelid=$1
        """, oid)
        for row in rows:
            cols = tuple(row["columns"])
            is_key = row["contype"] in ("p", "u")
            valid_index = (not is_key or (
                all(row[item] for item in ("indisvalid", "indisready", "indisunique", "indimmediate"))
                and not row["partial"] and not row["expression"]
                and row["indnatts"] == len(cols) and row["indnkeyatts"] == len(cols)))
            if (row["condeferrable"] or row["condeferred"] or not row["convalidated"]
                    or not valid_index):
                raise SchemaCompatibilityError("incompatible schema")
            if row["contype"] == "f":
                if row["referenced_schema"] != "public":
                    raise SchemaCompatibilityError("incompatible schema")
                item = (table, "f", cols, row["referenced_table"],
                        tuple(row["ref_columns"]), row["confdeltype"])
            else:
                item = (table, row["contype"], cols, None, (), None)
            if item in actual_constraints:
                raise SchemaCompatibilityError("incompatible schema")
            actual_constraints.add(item)
    if actual_constraints != expected_constraints:
        raise SchemaCompatibilityError("incompatible schema")
    if OBSERVATION_TABLE in names:
        await check_observation_schema(conn)
    if "schema_migrations" in names:
        _, versions = await ledger(conn, allow_bootstrap=True)
        if OBSERVATION_REVISION_VERSION in versions and OBSERVATION_TABLE not in names:
            raise SchemaCompatibilityError("incompatible schema")


async def check_runtime_compatibility(conn, *, require_observation=True):
    """Validate current structure and history using catalog reads only."""
    await check_application_schema(
        conn, ledger_optional=False,
        allow_missing_observations=not require_observation)
    await admin(conn, required_updated=True)
    await ledger(conn, allow_bootstrap=True, require_revision=True,
                 require_current=True, require_observation=require_observation)
