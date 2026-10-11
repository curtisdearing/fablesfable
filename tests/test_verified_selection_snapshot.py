import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_verified_snapshot_keeps_rejected_items_out_and_quotes_present():
 p=ROOT/'data/research_snapshots/2026-week-5/reviewed-selection-snapshot'
 cards=json.loads((p/'game_cards.json').read_text())
 reviews=[c['analyst_review'] for c in cards if c.get('analyst_review')]
 retained=[s for r in reviews for s in r['recommended_props']]
 assert len(retained)==10
 assert sum(r['best_selection']['status']=='manual_candidate_recheck_required' for r in reviews)==9
 assert sum(r['best_selection']['status']=='pass_pending_status' for r in reviews)==5
 public=[s for r in reviews for s in r['recommended_props']+r['conditional_props']+[r['best_selection']]]
 for s in public:
  assert not any(n in s['selection'] for n in ('Jayden Daniels','Tony Pollard','Malik Willis','Bijan Robinson'))
 for s in retained:
  assert s['verification_verdict']=='ACCEPT_AS_CONDITIONAL_ANALYST'
  assert all(s.get(k) for k in ('book','price','captured_at','why','risk'))
 for c in cards:
  if c.get('status')=='completed':continue
  assert all(m.get('provider')=='DraftKings' and m.get('captured_at') for m in c['game_markets'])
