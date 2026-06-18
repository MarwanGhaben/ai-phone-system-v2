import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_TEMPLATE = REPOSITORY_ROOT / "templates" / "dashboard.html"
DASHBOARD_SCHEMA = REPOSITORY_ROOT / "services" / "dashboard" / "db_init.sql"


def test_api_records_are_not_rendered_through_inner_html() -> None:
    template = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")

    assert not re.search(r"\.innerHTML\s*=\s*data\.", template)
    assert not re.search(r"\.innerHTML\s*=\s*\w+\.map\(", template)


def test_caller_actions_do_not_embed_values_in_inline_javascript() -> None:
    template = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")

    assert "onclick=\"editCaller('${caller.phone}'" not in template
    assert "onclick=\"deleteCaller('${caller.phone}')\"" not in template


def test_dashboard_schema_does_not_seed_an_administrator() -> None:
    schema = DASHBOARD_SCHEMA.read_text(encoding="utf-8")

    assert "INSERT INTO admin_users" not in schema
    assert "admin123" not in schema
