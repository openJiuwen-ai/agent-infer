#!/bin/sh
# Start as root, then switch to the UID that owns the working directory.
# That directory is the Buildkite checkout bind-mount, so files pytest writes
# stay owned by the agent and the next checkout cleanup can delete them.
# The image does not hardcode a UID: each agent machine can use its own.
set -eu

if [ "$(id -u)" != "0" ]; then
  exec "$@"
fi

# DSH functional stays root so the step can create test-user and pytest can
# `sudo -u test-user`. Other steps still drop to the checkout owner.
if [ "${AGENTINFER_CI_AS_ROOT:-}" = "1" ]; then
  # /tmp/ci-home is created in the image (Dockerfile). Keep it world-writable so
  # test-user can use HOME=/tmp/ci-home when the step sudo-wraps the bench.
  chmod 1777 /tmp/ci-home
  hf_home="${HF_HOME:-/home/models}"
  hf_cache="${HF_HUB_CACHE:-$hf_home/hub}"
  mkdir -p "$hf_home" "$hf_cache"
  chmod 1777 "$hf_home" "$hf_cache"
  exec "$@"
fi

uid=$(stat -c '%u' .)
gid=$(stat -c '%g' .)

if [ "$uid" = "0" ]; then
  exec "$@"
fi

# Match on the numeric id. The name ciagent may already exist for a different
# id, and groupadd/useradd fail on a duplicate name even when the id is free.
if ! getent group "$gid" >/dev/null 2>&1; then
  group_name=ciagent
  if getent group "$group_name" >/dev/null 2>&1; then
    group_name="ciagent${gid}"
  fi
  groupadd -g "$gid" -o "$group_name"
fi
if ! getent passwd "$uid" >/dev/null 2>&1; then
  user_name=ciagent
  if getent passwd "$user_name" >/dev/null 2>&1; then
    user_name="ciagent${uid}"
  fi
  useradd -u "$uid" -g "$gid" -o -M -d /tmp/ci-home -s /bin/bash "$user_name"
fi

name=$(getent passwd "$uid" | cut -d: -f1)
for extra in video render; do
  if getent group "$extra" >/dev/null 2>&1; then
    usermod -aG "$extra" "$name"
  fi
done

mkdir -p /tmp/ci-home
# Image builds run as root with HOME=/tmp/ci-home and leave root-owned
# subdirectories such as .cache. The process later drops privileges, so the
# whole tree must be owned by that user or flashinfer cannot create its workspace.
chown -R "$uid:$gid" /tmp/ci-home

# HF_HOME is a host bind mount. Own the directories so the dropped user can
# create cache entries. Do not chown -R: that would rewrite shared model files.
hf_home="${HF_HOME:-/home/models}"
hf_cache="${HF_HUB_CACHE:-$hf_home/hub}"
mkdir -p "$hf_home" "$hf_cache"
chown "$uid:$gid" "$hf_home" "$hf_cache"
chmod 1777 "$hf_home" "$hf_cache"
export HOME=/tmp/ci-home
export USER="$name"
export LOGNAME="$name"

exec python3 -c '
import os, pwd, sys
uid, gid = int(sys.argv[1]), int(sys.argv[2])
name = pwd.getpwuid(uid).pw_name
os.initgroups(name, gid)
os.setgid(gid)
os.setuid(uid)
os.execvp(sys.argv[3], sys.argv[3:])
' "$uid" "$gid" "$@"
