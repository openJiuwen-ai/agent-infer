#!/usr/bin/env bash
# Ship exactly one committed vllm-evolve tree to the remote workspace.
set -euo pipefail

remote="${VE_REMOTE:-gpu-host}"
workspace="${VE_REMOTE_WORKSPACE:-/workspace/vllm-evolve}"

repo_root="$(git rev-parse --show-toplevel)"
git_sha="$(git -C "${repo_root}" rev-parse HEAD)"
branch="$(git -C "${repo_root}" branch --show-current)"

if ! git -C "${repo_root}" diff --quiet || ! git -C "${repo_root}" diff --cached --quiet; then
  echo "ERROR: tracked changes must be committed before remote synchronization" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "${tmp_dir}"' EXIT
archive="${tmp_dir}/vllm-evolve-${git_sha}.tar"
git -C "${repo_root}" archive --format=tar --output="${archive}" "${git_sha}"
if command -v sha256sum >/dev/null 2>&1; then
  archive_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
else
  archive_sha256="$(shasum -a 256 "${archive}" | awk '{print $1}')"
fi

remote_staging="${workspace}/staging/source_${git_sha}_${archive_sha256}.tar"
remote_release="${workspace}/repo/releases/${git_sha}"
remote_current="${workspace}/repo/current"

ssh -o BatchMode=yes "${remote}" \
  "test -d $(printf '%q' "${workspace}") && test -w $(printf '%q' "${workspace}") && \
   mkdir -p $(printf '%q' "${workspace}/staging") $(printf '%q' "${workspace}/repo/releases")"
scp -q -o BatchMode=yes "${archive}" "${remote}:${remote_staging}"

ssh -o BatchMode=yes "${remote}" bash -s -- \
  "${remote_staging}" "${remote_release}" "${remote_current}" \
  "${git_sha}" "${branch}" "${archive_sha256}" <<'REMOTE'
set -euo pipefail
archive="$1"
release="$2"
current="$3"
git_sha="$4"
branch="$5"
expected_archive_sha="$6"

actual_archive_sha="$(sha256sum "${archive}" | awk '{print $1}')"
if [[ "${actual_archive_sha}" != "${expected_archive_sha}" ]]; then
  echo "ERROR: source archive SHA mismatch" >&2
  exit 3
fi

if [[ -e "${release}" ]]; then
  if [[ ! -f "${release}/.ve_source.json" ]] ||
     ! grep -q "\"git_sha\": \"${git_sha}\"" "${release}/.ve_source.json"; then
    echo "ERROR: refusing to overwrite existing release ${release}" >&2
    exit 4
  fi
else
  mkdir "${release}"
  tar -xf "${archive}" -C "${release}"
  cat > "${release}/.ve_source.json" <<EOF
{
  "git_sha": "${git_sha}",
  "branch": "${branch}",
  "archive_sha256": "${expected_archive_sha}"
}
EOF
fi

ln -sfn "${release}" "${current}.next"
mv -Tf "${current}.next" "${current}"
printf 'REMOTE_REPO=%s\nGIT_SHA=%s\nARCHIVE_SHA256=%s\n' \
  "${current}" "${git_sha}" "${expected_archive_sha}"
REMOTE
