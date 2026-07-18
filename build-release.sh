#!/usr/bin/env bash

# Validate and publish the version declared in inat_visualizer_version.py.
# Pushing the version tag triggers .github/workflows/release.yml, which builds
# and publishes the Windows, macOS, and Linux release artifacts.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./build-release.sh [--dry-run]

Validate main, read the application version from the codebase, and push the
matching vX.Y.Z tag to trigger the GitHub Build and Release workflow.

Options:
  --dry-run  Run all local and remote checks without creating or pushing a tag.
  -h, --help Show this help text.
EOF
}

dry_run=false
case "${1:-}" in
    "") ;;
    --dry-run) dry_run=true ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

script_dir="$(CDPATH='' cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [ -x ".venv/bin/python" ]; then
    python_bin=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
elif command -v python >/dev/null 2>&1; then
    python_bin="python"
else
    echo "Error: Python is required to read the application version." >&2
    exit 1
fi

version="$($python_bin -c 'from inat_visualizer_version import __version__; print(__version__)')"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid release version '$version'; expected X.Y.Z." >&2
    exit 1
fi
tag="v$version"

if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "Error: The working tree is not clean. Commit or remove pending changes first." >&2
    git status --short >&2
    exit 1
fi

branch="$(git symbolic-ref --quiet --short HEAD || true)"
if [ "$branch" != "main" ]; then
    echo "Error: Releases must be created from main; current branch is '${branch:-detached HEAD}'." >&2
    exit 1
fi

if ! grep -Fq "## [$version]" CHANGELOG.md; then
    echo "Error: CHANGELOG.md does not contain a section for $version." >&2
    exit 1
fi

"$python_bin" -m unittest -q test_version_metadata

echo "Fetching origin to verify that main and release tags are current..."
git fetch --quiet origin --tags

local_commit="$(git rev-parse HEAD)"
remote_commit="$(git rev-parse origin/main)"
if [ "$local_commit" != "$remote_commit" ]; then
    echo "Error: Local main does not match origin/main." >&2
    echo "  local:  $local_commit" >&2
    echo "  remote: $remote_commit" >&2
    exit 1
fi

if git rev-parse --quiet --verify "refs/tags/$tag" >/dev/null; then
    echo "Error: Tag $tag already exists. Bump the code version and changelog first." >&2
    exit 1
fi

echo "Release checks passed for $tag at commit $local_commit."
if [ "$dry_run" = true ]; then
    echo "Dry run complete; no tag was created or pushed."
    exit 0
fi

git tag -a "$tag" -m "Release $tag"
if ! git push origin "$tag"; then
    echo "Error: The tag was created locally but could not be pushed." >&2
    echo "After correcting the push problem, run: git push origin $tag" >&2
    exit 1
fi

echo "Release build triggered for $tag."
echo "Track it at:"
echo "https://github.com/AlanRockefeller/inat.visualizer.py/actions/workflows/release.yml"
