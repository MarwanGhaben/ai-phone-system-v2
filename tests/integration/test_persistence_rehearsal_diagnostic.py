"""Private release logs must never escape the allowlisted classifier."""
import importlib.util
import json
from pathlib import Path


def test_classifier_reports_error_categories_without_private_text():
    path = Path(__file__).resolve().parents[2] / 'scripts/inspect-persistence-rehearsal.py'
    spec = importlib.util.spec_from_file_location('rehearsal_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = 'SECRET-caller@example.invalid-and-password'
    text = ('settings=' + private + '\npg_restore: error: ' + private
            + '\nERROR: role "' + private + '" does not exist'
            + '\nSchemaCompatibilityError: ' + private)
    result = module.classify(text)
    assert [row['category'] for row in result] == [
        'restore_error', 'role_missing', 'schema_contract_rejected']
    assert private not in json.dumps(result)
    assert module.classify(private) == []
