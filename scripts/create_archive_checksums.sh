#!/bin/bash
# Create a deterministic SHA-256 manifest for the local experiment archive.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCHIVE_DIR="${PROJECT_DIR}/experiment_archive"
MANIFEST="${ARCHIVE_DIR}/SHA256SUMS"

if [[ ! -d "${ARCHIVE_DIR}" ]]; then
    echo "Archive directory not found: ${ARCHIVE_DIR}" >&2
    exit 1
fi

cd "${ARCHIVE_DIR}"
find . -type f ! -name 'SHA256SUMS' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${MANIFEST}"

echo "Created ${MANIFEST}"
echo "Files: $(wc -l < "${MANIFEST}")"
sha256sum "${MANIFEST}"
