export BASE_PATH=$(pwd) # Current directory
VECTOR_STORE="full"
# VECTOR_STORE="reduced"

NVME_TARGET="/opt/dlami/nvme/FEVER-8-Shared-Task/data_store/vector_store"
SYMLINK_PATH="${BASE_PATH}/data_store/vector_store"

if [ ! -d "$SYMLINK_PATH" ]; then
    # Ensure parent directory exists
    mkdir -p "$(dirname "$SYMLINK_PATH")"

    # Ensure NVMe target directory exists
    mkdir -p "$NVME_TARGET"

    # Download and extract directly into NVMe-backed storage
    curl -O "https://fever8-aic.s3.eu-west-2.amazonaws.com/${VECTOR_STORE}.tar.zst"
    tar --zstd -xvf "${VECTOR_STORE}.tar.zst" --strip-components=1 -C "$NVME_TARGET"
    rm -f "${VECTOR_STORE}.tar.zst"

    # Create symlink
    ln -s "$NVME_TARGET" "$SYMLINK_PATH"
fi