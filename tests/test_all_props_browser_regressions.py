import re
import shutil
import subprocess
from nflvalue import all_props


def test_rendered_browser_script_parses(tmp_path):
    node = shutil.which('node')
    if not node:
        import pytest
        pytest.skip('node unavailable')
    payload = {'state':'ready','cards':[], 'rows':[], 'counts':all_props._counts([],[])}
    scripts = re.findall(r'<script>(.*?)</script>', all_props.render_page(payload), re.S)
    assert scripts
    for n, script in enumerate(scripts):
        path=tmp_path/f'ui-{n}.js';path.write_text(script)
        subprocess.run([node,'--check',str(path)],check=True,capture_output=True)


def test_categorical_td_offer_and_placeholder_count():
    row={'player':'Real Player','side':'yes','line':None,'book':'Book','odds':120,'captured_at':'2026-10-10T17:00:00Z'}
    assert all_props._has_offer(row)
    gap={**row,'player':'No named player','raw_market_row_type':'coverage_gap'}
    assert not all_props._has_offer(gap)
    assert all_props._counts([], [all_props._normalise_row(row),all_props._normalise_row(gap)])['unique_athletes']==1


def test_et_clock_and_source_scheme():
    assert '4:05 PM ET' in all_props._kickoff('2026-10-11T20:05:00Z')
    assert not all_props._source_url('javascript:alert(1)')
    assert all_props._source_url('https://www.espn.com/')
