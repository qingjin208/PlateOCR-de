#!/usr/bin/env bash
# Build both iOS destinations and run the package regression tests on an Apple host.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$(mktemp -d -t plateocr-validation)"
echo "Validation logs and build products: ${build_root}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This check requires macOS with Xcode and the iOS SDK installed." >&2
    exit 1
fi

for destination in 'generic/platform=iOS' 'generic/platform=iOS Simulator'; do
    case "${destination}" in
        'generic/platform=iOS') label=device ;;
        *) label=simulator ;;
    esac
    log_path="${build_root}/${label}.log"
    echo "Building ${label}..."
    if ! xcodebuild -project "${project_root}/LicensePlateOCR.xcodeproj" \
        -scheme LicensePlateOCR -destination "${destination}" \
        -derivedDataPath "${build_root}/${label}" CODE_SIGNING_ALLOWED=NO build >"${log_path}" 2>&1; then
        tail -n 80 "${log_path}"
        echo "Build failed. Full log: ${log_path}" >&2
        exit 1
    fi
    echo "PASS: ${label} build"
done

echo "Running Swift Package tests..."
if ! swift test --package-path "${project_root}" >"${build_root}/swift-tests.log" 2>&1; then
    tail -n 80 "${build_root}/swift-tests.log"
    exit 1
fi
echo "PASS: Apple builds and package tests. Physical-device accuracy and energy checks remain separate."
