"""Read-only structural compatibility contract shared by startup and migrations."""
from __future__ import annotations
import json
import re

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
NOTIFICATION_REVISION_VERSION = "0004"
NOTIFICATION_REVISION_CHECKSUM = "a1c66c7705adbd49918966c7d41d8bbbc5e80d3d5d0534fe641fd15579731a9e"
NOTIFICATION_STATE_TABLE = "booking_notification_reconciliation"
NOTIFICATION_OUTBOX_TABLE = "booking_notification_outbox"
NOTIFICATION_CONTRACT_TABLE = "booking_notification_contract"
NOTIFICATION_TABLES = frozenset((NOTIFICATION_STATE_TABLE, NOTIFICATION_OUTBOX_TABLE,
                                 NOTIFICATION_CONTRACT_TABLE))
OPERATION_REVISION_VERSION = "0005"
OPERATION_REVISION_CHECKSUM = "cf9c9e2cdc53569a59833235efa98633b5f7b233f98549a8b2f1b16614866100"
OPERATION_TABLE = "booking_operations"
OPERATION_CONTRACT_TABLE = "booking_operation_contract"
OPERATION_TABLES = frozenset((OPERATION_TABLE, OPERATION_CONTRACT_TABLE))


class SchemaCompatibilityError(Exception):
    """A fixed, safe compatibility classification."""


_VARCHAR_LITERAL_ARRAY_CAST = re.compile(
    r"\(ARRAY\[((?:'[a-z_]+'::character varying)(?:, '[a-z_]+'::character varying)*)\]\)::text\[\]")


def check_definitions_match(baseline, actual, *, allow_operation_fk_visibility=False):
    """Allow narrowly proven PostgreSQL dump/restore serialization differences.

    Casting an unbounded varchar literal array to text[] is equivalent to casting
    each of its literal elements to text. pg_restore reparses the saved expression
    into the latter form. It also flattens the leading BETWEEN expansion in the
    notification missing-count conjunction. Keep all values, order, operators
    and other text exact; this is not a general SQL equivalence normalizer.
    """
    if (not isinstance(baseline, dict) or not isinstance(actual, dict)
            or baseline.keys() != actual.keys()
            or any(not isinstance(v, str) for v in (*baseline.values(), *actual.values()))):
        return False

    def canonical(value):
        value = _VARCHAR_LITERAL_ARRAY_CAST.sub(
            lambda match: 'ARRAY[' + ', '.join(
                '(' + item + ')::text' for item in match.group(1).split(', ')) + ']',
            value)
        nested = 'CHECK ((((missing_count >= 0) AND (missing_count <= 2)) AND '
        flat = 'CHECK (((missing_count >= 0) AND (missing_count <= 2) AND '
        # Restrict to this exact leading AND context: never strip arbitrary
        # parentheses around OR/NOT or change a bound/column/operator.
        if value.startswith(nested):
            value = flat + value[len(nested):]
        # pg_restore also flattens the first BETWEEN expansion in 0005's
        # identity-length conjunction, without changing its bound or operator.
        nested_identity = ("CHECK ((((length(tenant_id) >= 1) AND "
                           "(length(tenant_id) <= 255)) AND ")
        flat_identity = ("CHECK (((length(tenant_id) >= 1) AND "
                         "(length(tenant_id) <= 255) AND ")
        if value.startswith(nested_identity):
            value = flat_identity + value[len(nested_identity):]
        return value

    operation_fk = "booking_operations_booking_id_fkey"
    qualified = "FOREIGN KEY (booking_id) REFERENCES public.bookings(id)"
    visible = "FOREIGN KEY (booking_id) REFERENCES bookings(id)"
    for key in baseline:
        left, right = canonical(baseline[key]), canonical(actual[key])
        if left == right:
            continue
        # pg_get_constraintdef renders this one FK according to search_path.
        # Only these exact strings are equivalent; catalog checks below verify
        # referenced OID, column, actions and deferrability independently.
        if (not allow_operation_fk_visibility or key != operation_fk
                or {left, right} != {qualified, visible}):
            return False
    return True


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
NOTIFICATION_STATE_COLUMNS = _parse_spec(
    "booking_id:int4! snapshot_tenant_id:text! snapshot_business_id:text! "
    "snapshot_provider_id:varchar:255! snapshot_start:timestamptz! "
    "snapshot_version:int4! enrolled_at:timestamptz last_present_at:timestamptz "
    "last_evidence_started_at:timestamptz "
    "first_missing_at:timestamptz last_missing_at:timestamptz missing_count:int4! "
    "last_result:varchar:20 disposition:varchar:32! reason:varchar:40 "
    "transitioned_at:timestamptz last_error_category:varchar:32 updated_at:timestamptz!"
)
NOTIFICATION_OUTBOX_COLUMNS = _parse_spec(
    "id:uuid! booking_id:int4! snapshot_version:int4! snapshot_tenant_id:text! "
    "snapshot_business_id:text! snapshot_provider_id:varchar:255! "
    "snapshot_start:timestamptz! event_key:text! kind:varchar:20! "
    "due_at:timestamptz! next_attempt_at:timestamptz! state:varchar:20! "
    "attempts:int4! retry_count:int4! recipient:varchar:50! "
    "language:varchar:10! consultant:varchar:255! body:text! error_category:varchar:32 "
    "provider_message_id:varchar:255 claim_token:uuid dispatch_started_at:timestamptz "
    "last_attempt_at:timestamptz accepted_at:timestamptz created_at:timestamptz! "
    "updated_at:timestamptz!"
)
NOTIFICATION_CONTRACT_COLUMNS = _parse_spec("singleton:int4! check_definitions:jsonb!")
OPERATION_COLUMNS = _parse_spec(
    "operation_id:uuid! tenant_id:text! business_id:text! staff_id:text! "
    "service_id:text! action:text! proposal_id:text! proposal_revision:int8! "
    "payload_hash:text! payload_snapshot:jsonb! confirmation_identity:jsonb! "
    "confirmed_at:timestamptz! issued_at:timestamptz! expires_at:timestamptz! "
    "admitted_at:timestamptz! "
    "starts_at:timestamptz! ends_at:timestamptz! pre_buffer:interval! "
    "post_buffer:interval! claim_span:tstzrange! state:text! fence:int8! "
    "dispatch_count:int2! owner_token:uuid lease_until:timestamptz "
    "ownership_started_at:timestamptz "
    "provider_id:text receipt_provider_id:text booking_id:int4 verified_snapshot:jsonb "
    "settlement_source:text settlement_observed_at:timestamptz "
    "created_at:timestamptz! updated_at:timestamptz!"
)
OPERATION_CONTRACT_COLUMNS = _parse_spec("singleton:int4! constraint_definitions:jsonb!")


async def check_operation_schema(conn):
    """Reject 0005 structural, protective-constraint, index and extension drift."""
    extension = await conn.fetchrow("""
        SELECT n.nspname,e.extname FROM pg_catalog.pg_extension e
        JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace
        WHERE e.extname='btree_gist'
    """)
    if extension is None or extension["nspname"] != "public":
        raise SchemaCompatibilityError("incompatible schema")
    expected_constraints = {
        "booking_operations_pkey": ("p", ("operation_id",)),
        "booking_operations_intent_key": ("u", ("tenant_id", "business_id", "action",
                                                 "proposal_id", "proposal_revision")),
        "booking_operations_receipt_identity_key": ("u", ("tenant_id", "business_id",
                                                            "receipt_provider_id")),
        "booking_operations_booking_link_key": ("u", ("booking_id",)),
        "booking_operations_booking_id_fkey": ("f", ("booking_id",)),
        "booking_operations_staff_exclusion": ("x", ("tenant_id", "business_id",
                                                      "staff_id", "claim_span")),
        **{name: ("c", ()) for name in (
            "booking_operations_action_check", "booking_operations_identity_check",
            "booking_operations_payload_check", "booking_operations_clock_check",
            "booking_operations_interval_check", "booking_operations_state_check")},
    }
    oid = await relation(conn, OPERATION_TABLE)
    actual = await columns(conn, oid)
    check_columns(actual, OPERATION_COLUMNS)
    defaults = {"state": "'pending'::text", "fence": "0", "dispatch_count": "0",
                "created_at": "CURRENT_TIMESTAMP", "updated_at": "CURRENT_TIMESTAMP"}
    if any(row["default_expr"] != defaults.get(name) for name, row in actual.items()):
        raise SchemaCompatibilityError("incompatible schema")
    rows = await conn.fetch("""
        SELECT k.conname,k.contype::text,k.condeferrable,k.condeferred,k.convalidated,
               k.connoinherit,pg_catalog.pg_get_constraintdef(k.oid,false) AS definition,
               k.confrelid,k.confupdtype::text,k.confdeltype::text,k.confmatchtype::text,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.conkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=k.conrelid AND a.attnum=u.num ORDER BY u.ord) AS columns,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(k.confkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=k.confrelid AND a.attnum=u.num ORDER BY u.ord) AS reference_columns
        FROM pg_catalog.pg_constraint k WHERE k.conrelid=$1
    """, oid)
    if len(rows) != len(expected_constraints) or {r["conname"] for r in rows} != set(expected_constraints):
        raise SchemaCompatibilityError("incompatible schema")
    definitions = {}
    bookings_oid = await relation(conn, "bookings")
    for row in rows:
        kind, key = expected_constraints[row["conname"]]
        if (row["contype"] != kind or row["condeferrable"] or row["condeferred"]
                or not row["convalidated"] or (kind == "c" and row["connoinherit"])
                or (kind != "c" and tuple(row["columns"]) != key)):
            raise SchemaCompatibilityError("incompatible schema")
        if row["conname"] == "booking_operations_booking_id_fkey" and (
                row["confrelid"] != bookings_oid
                or tuple(row["reference_columns"]) != ("id",)
                or row["confupdtype"] != "a" or row["confdeltype"] != "a"
                or row["confmatchtype"] != "s"):
            raise SchemaCompatibilityError("incompatible schema")
        definitions[row["conname"]] = row["definition"]
    indexes = await conn.fetch("""
        SELECT c.relname,i.indisvalid,i.indisready,i.indisunique,i.indimmediate,
               i.indpred IS NOT NULL AS partial,i.indexprs IS NOT NULL AS expression,
               am.amname,
               ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(i.indkey)
                     WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                     ON a.attrelid=i.indrelid AND a.attnum=u.num ORDER BY u.ord) AS columns
        FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid
        JOIN pg_catalog.pg_class t ON t.oid=i.indrelid
        JOIN pg_catalog.pg_am am ON am.oid=c.relam
        WHERE i.indrelid=$1
    """, oid)
    expected_indexes = {"booking_operations_pkey", "booking_operations_intent_key",
                        "booking_operations_receipt_identity_key", "booking_operations_booking_link_key",
                        "booking_operations_staff_exclusion", "booking_operations_unresolved_idx"}
    if len(indexes) != len(expected_indexes) or {r["relname"] for r in indexes} != expected_indexes:
        raise SchemaCompatibilityError("incompatible schema")
    for row in indexes:
        name = row["relname"]
        expected_columns = {
            "booking_operations_pkey": ("operation_id",),
            "booking_operations_intent_key": ("tenant_id", "business_id", "action",
                                              "proposal_id", "proposal_revision"),
            "booking_operations_receipt_identity_key": ("tenant_id", "business_id",
                                                        "receipt_provider_id"),
            "booking_operations_booking_link_key": ("booking_id",),
            "booking_operations_staff_exclusion": ("tenant_id", "business_id", "staff_id",
                                                   "claim_span"),
            "booking_operations_unresolved_idx": ("state", "lease_until", "operation_id"),
        }
        if (not row["indisvalid"] or not row["indisready"] or not row["indimmediate"]
                or row["expression"] or row["partial"] != (name == "booking_operations_staff_exclusion")
                or tuple(row["columns"]) != expected_columns[name]
                or row["indisunique"] != (name in ("booking_operations_pkey",
                                                  "booking_operations_intent_key",
                                                  "booking_operations_receipt_identity_key",
                                                  "booking_operations_booking_link_key"))
                or row["amname"] != ("gist" if name == "booking_operations_staff_exclusion" else "btree")):
            raise SchemaCompatibilityError("incompatible schema")
    contract_oid = await relation(conn, OPERATION_CONTRACT_TABLE)
    contract_columns = await columns(conn, contract_oid)
    check_columns(contract_columns, OPERATION_CONTRACT_COLUMNS)
    if any(row["default_expr"] is not None for row in contract_columns.values()):
        raise SchemaCompatibilityError("incompatible schema")
    contract_rules = await conn.fetch("""
        SELECT conname,contype::text,convalidated,condeferrable,condeferred,
               pg_catalog.pg_get_constraintdef(oid,false) AS definition
        FROM pg_catalog.pg_constraint WHERE conrelid=$1
    """, contract_oid)
    if (len(contract_rules) != 2 or
            {row["conname"]: row["contype"] for row in contract_rules} != {
                "booking_operation_contract_pkey": "p",
                "booking_operation_contract_singleton_check": "c"}
            or any(not row["convalidated"] or row["condeferrable"] or row["condeferred"]
                   for row in contract_rules)):
        raise SchemaCompatibilityError("incompatible schema")
    singleton_check = next(row for row in contract_rules if row["contype"] == "c")
    if singleton_check["definition"] != "CHECK ((singleton = 1))":
        raise SchemaCompatibilityError("incompatible schema")
    baseline = await conn.fetch("SELECT singleton,constraint_definitions FROM public.booking_operation_contract")
    if len(baseline) != 1 or baseline[0]["singleton"] != 1:
        raise SchemaCompatibilityError("incompatible schema")
    recorded = baseline[0]["constraint_definitions"]
    if isinstance(recorded, str):
        try:
            recorded = json.loads(recorded)
        except ValueError:
            raise SchemaCompatibilityError("incompatible schema") from None
    if not check_definitions_match(recorded, definitions, allow_operation_fk_visibility=True):
        raise SchemaCompatibilityError("incompatible schema")


async def check_notification_schema(conn):
    """Validate 0004 shape, keys, indexes and exact migration-time CHECKs."""
    shapes = {
        NOTIFICATION_STATE_TABLE: NOTIFICATION_STATE_COLUMNS,
        NOTIFICATION_OUTBOX_TABLE: NOTIFICATION_OUTBOX_COLUMNS,
        NOTIFICATION_CONTRACT_TABLE: NOTIFICATION_CONTRACT_COLUMNS,
    }
    expected_constraints = {
        NOTIFICATION_STATE_TABLE: {
            'booking_notification_reconciliation_pkey': ('p', ('booking_id',)),
            'booking_notification_reconciliation_booking_id_fkey': ('f', ('booking_id',)),
            **{name: ('c', ()) for name in (
                'booking_notification_scope_check', 'booking_notification_missing_check',
                'booking_notification_disposition_check', 'booking_notification_result_check')},
        },
        NOTIFICATION_OUTBOX_TABLE: {
            'booking_notification_outbox_pkey': ('p', ('id',)),
            'booking_notification_outbox_booking_id_fkey': ('f', ('booking_id',)),
            'booking_notification_outbox_event_key_key': ('u', ('event_key',)),
            'booking_notification_outbox_identity_unique':
                ('u', ('booking_id', 'snapshot_version', 'kind')),
            **{name: ('c', ()) for name in (
                'booking_notification_outbox_scope_check',
                'booking_notification_outbox_kind_check',
                'booking_notification_outbox_state_check',
                'booking_notification_outbox_attempts_check',
                'booking_notification_outbox_claim_check',
                'booking_notification_outbox_acceptance_check')},
        },
        NOTIFICATION_CONTRACT_TABLE: {
            'booking_notification_contract_pkey': ('p', ('singleton',)),
            'booking_notification_contract_singleton_check': ('c', ()),
        },
    }
    expected_indexes = {
        NOTIFICATION_STATE_TABLE: {
            'booking_notification_reconciliation_pkey': ('booking_id',),
            'booking_notification_reconciliation_scope_idx':
                ('snapshot_tenant_id', 'snapshot_business_id', 'snapshot_provider_id')},
        NOTIFICATION_OUTBOX_TABLE: {
            'booking_notification_outbox_pkey': ('id',),
            'booking_notification_outbox_event_key_key': ('event_key',),
            'booking_notification_outbox_identity_unique':
                ('booking_id', 'snapshot_version', 'kind'),
            'booking_notification_outbox_due_idx':
                ('state', 'next_attempt_at', 'due_at', 'id'),
            'booking_notification_outbox_booking_idx': ('booking_id', 'state')},
        NOTIFICATION_CONTRACT_TABLE: {
            'booking_notification_contract_pkey': ('singleton',)},
    }
    definitions = {}
    for table, shape in shapes.items():
        oid = await relation(conn, table)
        actual = await columns(conn, oid)
        check_columns(actual, shape)
        if any(row['default_expr'] is not None for row in actual.values()):
            raise SchemaCompatibilityError('incompatible schema')
        rows = await conn.fetch("""
            SELECT k.conname,k.contype::text,k.condeferrable,k.condeferred,
                   k.convalidated,k.connoinherit,
                   pg_catalog.pg_get_constraintdef(k.oid,false) AS definition,
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
        expected = expected_constraints[table]
        if len(rows) != len(expected) or {r['conname'] for r in rows} != set(expected):
            raise SchemaCompatibilityError('incompatible schema')
        for row in rows:
            kind, key = expected[row['conname']]
            if (row['contype'] != kind or row['condeferrable'] or row['condeferred']
                    or not row['convalidated']):
                raise SchemaCompatibilityError('incompatible schema')
            if kind == 'c':
                if row['connoinherit']:
                    raise SchemaCompatibilityError('incompatible schema')
                definitions[row['conname']] = row['definition']
            elif tuple(row['columns']) != key:
                raise SchemaCompatibilityError('incompatible schema')
            if kind == 'f' and (row['referenced_schema'] != 'public'
                                or row['referenced_table'] != 'bookings'
                                or tuple(row['ref_columns']) != ('id',)
                                or row['confdeltype'] != 'r'):
                raise SchemaCompatibilityError('incompatible schema')
        indexes = await conn.fetch("""
            SELECT c.relname,i.indisvalid,i.indisready,i.indisunique,i.indimmediate,
                   i.indpred IS NOT NULL AS partial,i.indexprs IS NOT NULL AS expression,
                   i.indnatts,i.indnkeyatts,
                   ARRAY(SELECT a.attname::text FROM pg_catalog.unnest(i.indkey)
                         WITH ORDINALITY u(num,ord) JOIN pg_catalog.pg_attribute a
                         ON a.attrelid=i.indrelid AND a.attnum=u.num ORDER BY u.ord) AS columns
            FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid
            WHERE i.indrelid=$1
        """, oid)
        if len(indexes) != len(expected_indexes[table]) or {
                r['relname'] for r in indexes} != set(expected_indexes[table]):
            raise SchemaCompatibilityError('incompatible schema')
        for index in indexes:
            columns_expected = expected_indexes[table][index['relname']]
            unique_expected = index['relname'] in expected and expected[index['relname']][0] in ('p', 'u')
            if (tuple(index['columns']) != columns_expected
                    or index['indisunique'] != unique_expected
                    or not all(index[key] for key in ('indisvalid','indisready','indimmediate'))
                    or index['partial'] or index['expression']
                    or index['indnatts'] != len(columns_expected)
                    or index['indnkeyatts'] != len(columns_expected)):
                raise SchemaCompatibilityError('incompatible schema')
    rows = await conn.fetch(
        'SELECT singleton,check_definitions FROM public.booking_notification_contract')
    if len(rows) != 1 or rows[0]['singleton'] != 1:
        raise SchemaCompatibilityError('incompatible schema')
    baseline = rows[0]['check_definitions']
    if isinstance(baseline, str):
        try:
            baseline = json.loads(baseline)
        except ValueError:
            raise SchemaCompatibilityError('incompatible schema') from None
    if not check_definitions_match(baseline, definitions):
        raise SchemaCompatibilityError('incompatible schema')


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
    if not check_definitions_match(baseline, definitions):
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
                 require_current=False, require_observation=False,
                 require_notification=False):
    oid = await relation(conn, "schema_migrations", optional=True)
    if oid is None:
        if require_revision or require_current or require_observation or require_notification:
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
        NOTIFICATION_REVISION_VERSION: NOTIFICATION_REVISION_CHECKSUM,
        OPERATION_REVISION_VERSION: OPERATION_REVISION_CHECKSUM,
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
    if (NOTIFICATION_REVISION_VERSION in versions
            and OBSERVATION_REVISION_VERSION not in versions):
        raise SchemaCompatibilityError("incompatible schema")
    if (OPERATION_REVISION_VERSION in versions
            and NOTIFICATION_REVISION_VERSION not in versions):
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
    if (NOTIFICATION_REVISION_VERSION in versions and OBSERVATION_REVISION_VERSION in versions
            and applied[OBSERVATION_REVISION_VERSION] > applied[NOTIFICATION_REVISION_VERSION]):
        raise SchemaCompatibilityError("incompatible schema")
    if (OPERATION_REVISION_VERSION in versions and NOTIFICATION_REVISION_VERSION in versions
            and applied[NOTIFICATION_REVISION_VERSION] > applied[OPERATION_REVISION_VERSION]):
        raise SchemaCompatibilityError("incompatible schema")
    if require_notification and NOTIFICATION_REVISION_VERSION not in versions:
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
                                   allow_missing_notifications=True,
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
    if NOTIFICATION_TABLES.issubset(names):
        expected_names.update(NOTIFICATION_TABLES)
    elif not allow_missing_notifications:
        raise SchemaCompatibilityError("incompatible schema")
    if OPERATION_TABLES.issubset(names):
        expected_names.update(OPERATION_TABLES)
    if names != expected_names:
        raise SchemaCompatibilityError("incompatible schema")
    lock_names = sorted(TABLE_COLUMNS)
    if "schema_migrations" in names:
        lock_names.append("schema_migrations")
    if OBSERVATION_TABLE in names:
        lock_names.extend((OBSERVATION_TABLE, OBSERVATION_CONTROL_TABLE))
    if NOTIFICATION_TABLES.issubset(names):
        lock_names.extend(sorted(NOTIFICATION_TABLES))
    if OPERATION_TABLES.issubset(names):
        lock_names.extend(sorted(OPERATION_TABLES))
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
    if NOTIFICATION_TABLES.issubset(names):
        await check_notification_schema(conn)
    if OPERATION_TABLES.issubset(names):
        await check_operation_schema(conn)
    if "schema_migrations" in names:
        _, versions = await ledger(conn, allow_bootstrap=True)
        if OBSERVATION_REVISION_VERSION in versions and OBSERVATION_TABLE not in names:
            raise SchemaCompatibilityError("incompatible schema")
        if NOTIFICATION_REVISION_VERSION in versions and not NOTIFICATION_TABLES.issubset(names):
            raise SchemaCompatibilityError("incompatible schema")
        if NOTIFICATION_TABLES.issubset(names) and NOTIFICATION_REVISION_VERSION not in versions:
            raise SchemaCompatibilityError("incompatible schema")
        if OPERATION_REVISION_VERSION in versions and not OPERATION_TABLES.issubset(names):
            raise SchemaCompatibilityError("incompatible schema")
        if OPERATION_TABLES.issubset(names) and OPERATION_REVISION_VERSION not in versions:
            raise SchemaCompatibilityError("incompatible schema")


async def check_runtime_compatibility(conn, *, require_observation=True,
                                      require_notification=False):
    """Validate current structure and history using catalog reads only."""
    await check_application_schema(
        conn, ledger_optional=False,
        allow_missing_observations=not require_observation,
        allow_missing_notifications=not require_notification)
    await admin(conn, required_updated=True)
    await ledger(conn, allow_bootstrap=True, require_revision=True,
                 require_current=True, require_observation=require_observation)
    if require_notification:
        await ledger(conn, allow_bootstrap=True, require_notification=True)
