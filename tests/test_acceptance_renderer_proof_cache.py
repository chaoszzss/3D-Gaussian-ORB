"""Evidence caching must not skip a separately declared integration proof."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import acceptance_gate as gate
import measure_surface_quality as surface


def test_distinct_integration_proofs_are_validated_even_for_same_renderer(monkeypatch):
    calls = []
    def validate(report):
        calls.append(report['renderer_integration_proof']['sha256'])
        if calls[-1] == 'bad':
            raise ValueError('bad integration proof')
        return {'backend': report['backend'], 'assets': {}}
    monkeypatch.setattr(surface, 'validate_renderer_provenance', validate)
    report = dict(backend='hbb1-aa-depth-vjp-2dgs', renderer_provenance={'variant': 'aa-depth-vjp'},
                  renderer_integration_proof={'path': '/proof.json', 'sha256': 'good'})
    cache = {}
    gate.validate_backend(report, cache)
    gate.validate_backend(report, cache)
    report['renderer_integration_proof']['sha256'] = 'bad'
    with pytest.raises(ValueError, match='bad integration'):
        gate.validate_backend(report, cache)
    assert calls == ['good', 'bad']


def test_original_profile_identity_does_not_depend_on_optional_legacy_metadata():
    legacy = dict(backend='hbb1-original-2dgs')
    explicit = dict(backend='hbb1-original-2dgs', renderer_provenance={'variant': 'original'})
    assert gate.validate_backend(legacy) == gate.validate_backend(explicit)
    with pytest.raises(ValueError, match='mislabeled'):
        gate.validate_backend(dict(backend='hbb1-original-2dgs', renderer_provenance={'variant': 'aa-depth-vjp'}))
