"""Public reading must not require model or wagering writes (stdlib-only)."""
import hashlib
import json
import os
from pathlib import Path
import re
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _workflow_checker():
    """Extract the inline stdlib validator so fixtures exercise production code."""
    workflow = (ROOT / '.github/workflows/website.yml').read_text()
    match = re.search(
        r"          python3 - <<'PY'\n(?P<checker>.*?)\n          PY",
        workflow,
        re.DOTALL,
    )
    assert match, 'website workflow must contain its static artifact checker'
    return textwrap.dedent(match.group('checker'))


def _run_workflow_checker(project_root):
    previous_directory = Path.cwd()
    try:
        os.chdir(project_root)
        exec(
            compile(_workflow_checker(), 'website.yml inline checker', 'exec'),
            {'__name__': '__main__'},
        )
    finally:
        os.chdir(previous_directory)


def _sha256(contents):
    return hashlib.sha256(contents.encode()).hexdigest()


def _write_manifest(root, files):
    manifest = {
        'kind': 'saved-model-analysis',
        'approved_bets': 0,
        'files': {name: _sha256(contents) for name, contents in files.items()},
    }
    (root / 'publication.json').write_text(json.dumps(manifest))


def _public_fixture(tmp_path):
    """Create deliberately synthetic static files, not product publication data."""
    root = tmp_path / 'published-site'
    root.mkdir(parents=True)
    files = {
        'index.html': '<!doctype html><title>Synthetic index</title>',
        'best-bets.html': '<!doctype html><title>Synthetic best bets</title>',
        'history.html': '<!doctype html><title>Synthetic history</title>',
        'api/hub.json': '{"fixture": "synthetic"}',
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    _write_manifest(root, files)
    return root, files


def _declare(root, files, name, contents):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    files[name] = contents
    _write_manifest(root, files)


def test_static_website_can_deploy_without_model_workflow():
    path = ROOT / '.github/workflows/website.yml'
    assert path.is_file(), 'A separate static website publication path is required'
    text = path.read_text()
    assert re.search(r'branches: \[main\]', text)
    assert 'published-site/**' in text
    assert 'contents: read' in text
    assert 'name: github-pages' in text
    for forbidden in ['ODDS_API_KEY', 'DISCORD_WEBHOOK_URL', 'auto_weekly.py', 'state_store.py', 'model-state']:
        assert forbidden not in text
    assert 'actions/upload-pages-artifact@v4' in text
    assert 'actions/deploy-pages@v4' in text
    assert 'path: published-site' in text


def test_ui_only_push_does_not_run_model_and_legacy_deploy_cannot_overwrite_ui():
    text = (ROOT / '.github/workflows/live-weekly.yml').read_text()
    push = text.split('  push:', 1)[1].split('  workflow_dispatch:', 1)[0]
    assert 'paths-ignore:' in push
    for path in ['published-site/**', '.github/workflows/website.yml', '.github/workflows/live-weekly.yml', 'tests/test_readonly_website_workflow.py']:
        assert path in push
    deploy = text.split('  deploy-dashboard:', 1)[1]
    assert re.search(r'^    if: \$\{\{ false \}\}$', deploy, re.M)
    # Existing schedules and model execution are intentionally preserved.
    assert '  schedule:' in text
    assert 'run: python scripts/auto_weekly.py' in text


def test_static_checker_accepts_a_tiny_canonical_synthetic_fixture(tmp_path):
    _public_fixture(tmp_path)

    _run_workflow_checker(tmp_path)


def test_static_checker_accepts_each_supported_canonical_path_family(tmp_path):
    root, files = _public_fixture(tmp_path)
    for name, contents in {
        'dashboard.html': '<p>synthetic dashboard</p>',
        'history.json': '{"fixture": "synthetic history"}',
        'README.txt': 'synthetic documentation',
        'reports/index.json': '{"fixture": "synthetic report index"}',
        'reports/latest.html': '<p>synthetic latest report</p>',
        'reports/2026/week-22.html': '<p>synthetic weekly report</p>',
        'games/pages.json': '{"fixture": "synthetic game pages"}',
        'games/index.json': '{"fixture": "synthetic game index"}',
        'games/2026_01_NE_NYG.html': '<p>synthetic game page</p>',
        'assets/site.css': 'body {}',
        'assets/site.js': 'void 0;',
        'assets/logo.svg': '<svg/>',
        'assets/logo.png': 'synthetic png bytes',
        'assets/favicon.ico': 'synthetic ico bytes',
    }.items():
        _declare(root, files, name, contents)

    _run_workflow_checker(tmp_path)


def test_static_checker_rejects_a_symlinked_publication_root(tmp_path):
    target = tmp_path / 'real-publication-root'
    target.mkdir()
    _, files = _public_fixture(tmp_path / 'fixture')
    fixture_root = tmp_path / 'fixture' / 'published-site'
    for child in fixture_root.iterdir():
        child.rename(target / child.name)
    fixture_root.rmdir()
    (tmp_path / 'published-site').symlink_to(target, target_is_directory=True)

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


def test_static_checker_rejects_a_symlinked_manifest(tmp_path):
    root, _ = _public_fixture(tmp_path)
    external_manifest = tmp_path / 'synthetic-manifest.json'
    (root / 'publication.json').rename(external_manifest)
    (root / 'publication.json').symlink_to(external_manifest)

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


def test_static_checker_rejects_an_unreferenced_nested_directory_symlink(tmp_path):
    root, _ = _public_fixture(tmp_path)
    empty_target = root / 'synthetic-empty-directory'
    empty_target.mkdir()
    (root / 'nested-link').symlink_to(empty_target, target_is_directory=True)

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


def test_static_checker_rejects_a_declared_noncanonical_html_payload(tmp_path):
    root, files = _public_fixture(tmp_path)
    _declare(root, files, 'unapproved.html', '<p>synthetic unapproved payload</p>')

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


def test_static_checker_rejects_an_undeclared_payload(tmp_path):
    root, _ = _public_fixture(tmp_path)
    (root / 'unlisted.html').write_text('<p>synthetic undeclared payload</p>')

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


def test_static_checker_rejects_manifest_path_traversal(tmp_path):
    root, files = _public_fixture(tmp_path)
    external = tmp_path / 'outside.html'
    external.write_text('<p>synthetic outside payload</p>')
    files['../outside.html'] = external.read_text()
    _write_manifest(root, files)

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


def test_static_checker_rejects_an_arbitrary_root_text_file(tmp_path):
    root, files = _public_fixture(tmp_path)
    _declare(root, files, 'secret.txt', 'synthetic secret')

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)


@pytest.mark.parametrize('name', ['assets/secret.json', 'assets/secret.txt', 'site.css'])
def test_static_checker_rejects_nonstatic_asset_extensions(tmp_path, name):
    root, files = _public_fixture(tmp_path)
    _declare(root, files, name, 'synthetic disallowed asset')

    with pytest.raises(AssertionError):
        _run_workflow_checker(tmp_path)
