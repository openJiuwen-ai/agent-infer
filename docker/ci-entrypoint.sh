#!/bin/sh
# Start as root, then switch to the UID that owns the working directory.
# That directory is the Buildkite checkout bind-mount, so files pytest writes
# stay owned by the agent and the next checkout cleanup can delete them.
# The image does not hardcode a UID: each agent machine can use its own.
set -eu

if [ "$(id -u)" != "0" ]; then
  exec "$@"
fi

uid=$(stat -c '%u' .)
gid=$(stat -c '%g' .)

if [ "$uid" = "0" ]; then
  exec "$@"
fi

if ! getent group "$gid" >/dev/null 2>&1; then
  groupadd -g "$gid" -o ciagent
fi
if ! getent passwd "$uid" >/dev/null 2>&1; then
  useradd -u "$uid" -g "$gid" -o -M -d /tmp/ci-home -s /bin/bash ciagent
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
