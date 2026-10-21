# Mala launchd artifacts

Shell wrappers and `.plist` templates for launchd-triggered Mala jobs.
These are checked into the repo as templates so the install step is
reproducible.

## Current jobs

| Job | Wrapper | Plist | Schedule |
|---|---|---|---|
| Nightly regime matrix scout | `run_nightly_regime_matrix.sh` | `com.mala.nightly_regime_matrix.plist` | Daily 02:00 local time |

## Install (one-time per machine)

The `.plist` files in this directory use `HOME_PLACEHOLDER` as a
stand-in for `$HOME` so the template is portable. The install steps
substitute the real `$HOME` and copy the plist into
`~/Library/LaunchAgents/`.

```bash
cd "$(dirname "$0")"   # this directory
JOB=com.mala.nightly_regime_matrix

# 1. Make sure the wrapper is executable (one-time — git preserves the
#    bit, but if you created the file locally it might not be set).
chmod +x run_nightly_regime_matrix.sh

# 2. Substitute HOME into the plist and install it.
sed "s|HOME_PLACEHOLDER|$HOME|g" "$JOB.plist" \
    > "$HOME/Library/LaunchAgents/$JOB.plist"

# 3. Load it into launchd (the -w flag marks it enabled across reboots).
launchctl load -w "$HOME/Library/LaunchAgents/$JOB.plist"

# 4. Verify it's registered.
launchctl list | grep "$JOB"

# 5. Optional: trigger a one-off run right now to sanity-check the wrapper.
launchctl start "$JOB"

# Logs land in:
#   mala_v1/logs/nightly_regime_matrix.out.log
#   mala_v1/logs/nightly_regime_matrix.err.log
```

## Uninstall

```bash
JOB=com.mala.nightly_regime_matrix
launchctl unload "$HOME/Library/LaunchAgents/$JOB.plist"
rm "$HOME/Library/LaunchAgents/$JOB.plist"
```

## Notes

- **Laptop sleep:** launchd's `StartCalendarInterval` catches up a
  missed run when the laptop wakes. You don't need the machine awake at
  exactly 02:00 — if it's asleep, the next wake triggers the run.
- **`uv` on PATH:** the wrapper sets an explicit `PATH` because
  launchd hands jobs a minimal environment that usually doesn't include
  `~/.cargo/bin` or `~/.local/bin` where `uv` lives. If `uv=missing`
  shows up in the log, adjust the PATH line in the wrapper to match
  where your `uv` binary is installed.
- **Multiple machines:** the `HOME_PLACEHOLDER` substitution means the
  same template works on any macOS host. Re-run the install step on
  each machine.
