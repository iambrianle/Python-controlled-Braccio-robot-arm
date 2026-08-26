#!/usr/bin/env bash
set -euo pipefail

OUTPUT="braccio recording.mp4"
TARGET_MB=10

# ── Install ffmpeg if missing ────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found – installing via Homebrew…"
    brew install ffmpeg
fi

# Locate the .mov file via glob (handles Unicode/special chars in filename)
INPUT=$(ls -- *.mov 2>/dev/null | head -n1)
if [[ -z "$INPUT" ]]; then
    echo "Error: no .mov file found in $(pwd)"
    exit 1
fi
echo "Input file: $INPUT"

ORIGINAL_MB=$(du -m "$INPUT" | cut -f1)
echo "Original file: ${ORIGINAL_MB} MB"
echo "Target size:   ${TARGET_MB} MB"

# ── Calculate target bitrate for two-pass encode ─────────────────────────────
# ffprobe gives duration in seconds
DURATION=$(ffprobe -v error -select_streams v:0 \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$INPUT")

DURATION_INT=${DURATION%.*}
# target_bitrate (kbps) = (target_bytes * 8) / duration  – reserve ~64 kbps for audio
TARGET_KBPS=$(( (TARGET_MB * 1024 * 8) / DURATION_INT - 64 ))

if (( TARGET_KBPS < 100 )); then
    echo "Warning: target bitrate very low (${TARGET_KBPS} kbps). Consider raising TARGET_MB."
fi

echo "Duration: ${DURATION_INT}s  →  video bitrate target: ${TARGET_KBPS} kbps"
echo "Encoding (two-pass)…"

# ── Two-pass H.264 encode ────────────────────────────────────────────────────
# Pass 1 (analysis only, no output file)
ffmpeg -y -i "$INPUT" \
    -c:v libx264 -b:v "${TARGET_KBPS}k" \
    -vf "scale='min(1280,iw)':-2" \
    -pass 1 -an -f null /dev/null

# Pass 2 (final encode)
ffmpeg -y -i "$INPUT" \
    -c:v libx264 -b:v "${TARGET_KBPS}k" \
    -vf "scale='min(1280,iw)':-2" \
    -pass 2 \
    -c:a aac -b:a 64k \
    -movflags +faststart \
    "$OUTPUT"

# ── Cleanup two-pass log files ───────────────────────────────────────────────
rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree

RESULT_MB=$(du -m "$OUTPUT" | cut -f1)
echo ""
echo "Done! '$OUTPUT' → ${RESULT_MB} MB"
