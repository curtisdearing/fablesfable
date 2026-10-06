#!/bin/bash
# Install (or reinstall) the per-slot T-90 scheduler as a macOS LaunchAgent.
#   scripts/install_pregame_scheduler.sh [OPS_DIR]        default OPS_DIR=~/fablesfable-ops
#   scripts/install_pregame_scheduler.sh --uninstall
# Needs: git, python3 (stdlib only) and an authenticated `gh` on PATH. The agent ticks every
# 5 minutes while you are logged in and the Mac is awake; a slot that passes while the Mac
# sleeps is announced as MISSED on the next tick.
set -euo pipefail
LABEL=com.fablesfable.pregame-scheduler
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
if [ "${1:-}" = "--uninstall" ]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"; echo "uninstalled $LABEL"; exit 0
fi
OPS="${1:-$HOME/fablesfable-ops}"
RUNNER="$OPS/runner"
PY="$(command -v python3)"; GH="$(command -v gh)"
gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated"; exit 1; }
mkdir -p "$OPS/logs" "$OPS/state" "$OPS/receipts" "$HOME/Library/LaunchAgents"
[ -d "$RUNNER/.git" ] || git clone -q https://github.com/curtisdearing/fablesfable.git "$RUNNER"
git -C "$RUNNER" fetch -q origin main && git -C "$RUNNER" checkout -q --detach origin/main
[ -f "$RUNNER/scripts/pregame_scheduler.py" ] || { echo "scheduler not on origin/main yet"; exit 1; }
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string><string>$RUNNER/scripts/pregame_scheduler.py</string>
    <string>--ops-dir</string><string>$OPS</string><string>--repo-dir</string><string>$RUNNER</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>$(dirname "$GH"):$(dirname "$PY"):/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>AbandonProcessGroup</key><true/>
  <key>StandardOutPath</key><string>$OPS/logs/scheduler.out.log</string>
  <key>StandardErrorPath</key><string>$OPS/logs/scheduler.err.log</string>
</dict></plist>
PL
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "installed $LABEL -> $PLIST (ticks every 5 min; logs in $OPS/logs)"
"$PY" "$RUNNER/scripts/pregame_scheduler.py" --ops-dir "$OPS" --repo-dir "$RUNNER" --no-sync --dry-run | tail -1
