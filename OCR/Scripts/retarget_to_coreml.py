#!/usr/bin/env python3
"""Retarget the LicensePlateOCR Xcode project from ONNX Runtime (CocoaPods) to the
official UltralyticsYOLO Swift Package (Core ML pose).

What it does to LicensePlateOCR.xcodeproj/project.pbxproj:
  1. Adds an XCRemoteSwiftPackageReference for https://github.com/ultralytics/yolo-ios-app
     (up to next major from 8.9.18) and links the UltralyticsYOLO product to the app target.
  2. Removes every CocoaPods / onnxruntime trace: Pods.xcodeproj reference, the
     [CP] Check Pods Manifest.lock build phase, libPods / libonnxruntime reference
     proxies, the Pods target dependency and the Pods xcconfig references.

The project uses a PBXFileSystemSynchronizedRootGroup for LicensePlateOCR/, so adding
or deleting Swift files and resources needs no pbxproj edits - only this dependency
swap has to be done here.

Idempotent: re-running detects the package reference and exits without changes.
The original file is backed up once as project.pbxproj.bak.

Usage:
    python Scripts/retarget_to_coreml.py            # patch
    python Scripts/retarget_to_coreml.py --check    # report only, no writes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PBX = REPO_ROOT / "LicensePlateOCR.xcodeproj" / "project.pbxproj"
BACKUP = REPO_ROOT / "LicensePlateOCR.xcodeproj" / "project.pbxproj.bak"

REPO_URL = "https://github.com/ultralytics/yolo-ios-app"
MIN_VERSION = "8.9.18"

# Stable, unused 24-hex identifiers for the objects we add.
PKG_REF = "1A2B3C4D5E6F708192A3B4C5"
PROD_REF = "1A2B3C4D5E6F708192A3B4C6"
LINK_REF = "1A2B3C4D5E6F708192A3B4C7"

# ---- text we remove (must match the current file exactly) --------------------
REMOVE_LINES = [
    # Pods build file + its Frameworks phase entry
    "\t\t04F04279CA2B14773D17F672 /* libPods-LicensePlateOCR.a in Frameworks */ = {isa = PBXBuildFile; fileRef = 44737BEFAC7BC1D1BED5C969 /* libPods-LicensePlateOCR.a */; };\n",
    "\t\t\t\t04F04279CA2B14773D17F672 /* libPods-LicensePlateOCR.a in Frameworks */,\n",
    # file references
    "\t\t44737BEFAC7BC1D1BED5C969 /* libPods-LicensePlateOCR.a */ = {isa = PBXFileReference; explicitFileType = archive.ar; includeInIndex = 0; path = \"libPods-LicensePlateOCR.a\"; sourceTree = BUILT_PRODUCTS_DIR; };\n",
    "\t\tA1B2C3D4E5F607182930A5B6 /* Pods.xcodeproj */ = {isa = PBXFileReference; lastKnownFileType = \"wrapper.pb-project\"; name = Pods.xcodeproj; path = Pods/Pods.xcodeproj; sourceTree = \"<group>\"; };\n",
    "\t\tE5CEE3008D4FCDB9239C0720 /* Pods-LicensePlateOCR.debug.xcconfig */ = {isa = PBXFileReference; includeInIndex = 1; lastKnownFileType = text.xcconfig; name = \"Pods-LicensePlateOCR.debug.xcconfig\"; path = \"Target Support Files/Pods-LicensePlateOCR/Pods-LicensePlateOCR.debug.xcconfig\"; sourceTree = \"<group>\"; };\n",
    "\t\tE7E80A76A563AE2644F78404 /* Pods-LicensePlateOCR.release.xcconfig */ = {isa = PBXFileReference; includeInIndex = 1; lastKnownFileType = text.xcconfig; name = \"Pods-LicensePlateOCR.release.xcconfig\"; path = \"Target Support Files/Pods-LicensePlateOCR/Pods-LicensePlateOCR.release.xcconfig\"; sourceTree = \"<group>\"; };\n",
    # main group children
    "\t\t\t\t3C10B1554B03B5886E3D3A1E /* Pods */,\n",
    "\t\t\t\tA1B2C3D4E5F607182930A5B6 /* Pods.xcodeproj */,\n",
    "\t\t\t\t57E640F12E5A7D57C896CBE6 /* Frameworks */,\n",
    # target build phase / dependency
    "\t\t\t\t6EC052693C1ABD4570E06A08 /* [CP] Check Pods Manifest.lock */,\n",
    "\t\t\t\tA1B2C3D4E5F607182930A5B8 /* PBXTargetDependency */,\n",
    # projectReferences entry
    "\t\t\t\t{\n\t\t\t\t\tProductGroup = 37E9C4CE308F9FA000953685 /* Products */;\n\t\t\t\t\tProjectRef = A1B2C3D4E5F607182930A5B6 /* Pods.xcodeproj */;\n\t\t\t\t},\n",
]

REMOVE_BLOCKS = [
    # PBXContainerItemProxy entries
    "\t\t37E9C4D3308F9FA000953685 /* PBXContainerItemProxy */ = {\n\t\t\tisa = PBXContainerItemProxy;\n\t\t\tcontainerPortal = A1B2C3D4E5F607182930A5B6 /* Pods.xcodeproj */;\n\t\t\tproxyType = 2;\n\t\t\tremoteGlobalIDString = EFCD5A1058E9109AAD4F0B9F6ABE47E9;\n\t\t\tremoteInfo = \"onnxruntime-mobile-objc\";\n\t\t};\n",
    "\t\t37E9C4D5308F9FA000953685 /* PBXContainerItemProxy */ = {\n\t\t\tisa = PBXContainerItemProxy;\n\t\t\tcontainerPortal = A1B2C3D4E5F607182930A5B6 /* Pods.xcodeproj */;\n\t\t\tproxyType = 2;\n\t\t\tremoteGlobalIDString = 387CCBF166A71A9766F044D1CBF6F5B2;\n\t\t\tremoteInfo = \"Pods-LicensePlateOCR\";\n\t\t};\n",
    "\t\tA1B2C3D4E5F607182930A5B7 /* PBXContainerItemProxy */ = {\n\t\t\tisa = PBXContainerItemProxy;\n\t\t\tcontainerPortal = A1B2C3D4E5F607182930A5B6 /* Pods.xcodeproj */;\n\t\t\tproxyType = 1;\n\t\t\tremoteGlobalIDString = 9B248793ED5FDB77258CEAE14A0A5066;\n\t\t\tremoteInfo = \"Pods-LicensePlateOCR\";\n\t\t};\n",
    # PBXGroup entries
    "\t\t37E9C4CE308F9FA000953685 /* Products */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (\n\t\t\t\t37E9C4D4308F9FA000953685 /* libonnxruntime-mobile-objc.a */,\n\t\t\t\t37E9C4D6308F9FA000953685 /* libPods-LicensePlateOCR.a */,\n\t\t\t);\n\t\t\tname = Products;\n\t\t\tsourceTree = \"<group>\";\n\t\t};\n",
    "\t\t3C10B1554B03B5886E3D3A1E /* Pods */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (\n\t\t\t\tE5CEE3008D4FCDB9239C0720 /* Pods-LicensePlateOCR.debug.xcconfig */,\n\t\t\t\tE7E80A76A563AE2644F78404 /* Pods-LicensePlateOCR.release.xcconfig */,\n\t\t\t);\n\t\t\tpath = Pods;\n\t\t\tsourceTree = \"<group>\";\n\t\t};\n",
    "\t\t57E640F12E5A7D57C896CBE6 /* Frameworks */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (\n\t\t\t\t44737BEFAC7BC1D1BED5C969 /* libPods-LicensePlateOCR.a */,\n\t\t\t);\n\t\t\tname = Frameworks;\n\t\t\tsourceTree = \"<group>\";\n\t\t};\n",
    # PBXReferenceProxy entries
    "\t\t37E9C4D4308F9FA000953685 /* libonnxruntime-mobile-objc.a */ = {\n\t\t\tisa = PBXReferenceProxy;\n\t\t\tfileType = archive.ar;\n\t\t\tpath = \"libonnxruntime-mobile-objc.a\";\n\t\t\tremoteRef = 37E9C4D3308F9FA000953685 /* PBXContainerItemProxy */;\n\t\t\tsourceTree = BUILT_PRODUCTS_DIR;\n\t\t};\n",
    "\t\t37E9C4D6308F9FA000953685 /* libPods-LicensePlateOCR.a */ = {\n\t\t\tisa = PBXReferenceProxy;\n\t\t\tfileType = archive.ar;\n\t\t\tpath = \"libPods-LicensePlateOCR.a\";\n\t\t\tremoteRef = 37E9C4D5308F9FA000953685 /* PBXContainerItemProxy */;\n\t\t\tsourceTree = BUILT_PRODUCTS_DIR;\n\t\t};\n",
    # [CP] Check Pods Manifest.lock script phase
    "\t\t6EC052693C1ABD4570E06A08 /* [CP] Check Pods Manifest.lock */ = {\n\t\t\tisa = PBXShellScriptBuildPhase;\n",
    # PBXTargetDependency
    "\t\tA1B2C3D4E5F607182930A5B8 /* PBXTargetDependency */ = {\n\t\t\tisa = PBXTargetDependency;\n\t\t\tname = \"Pods-LicensePlateOCR\";\n\t\t\ttargetProxy = A1B2C3D4E5F607182930A5B7 /* PBXContainerItemProxy */;\n\t\t};\n",
]

# ---- text we add ------------------------------------------------------------
PKG_SECTION = f"""
/* Begin XCRemoteSwiftPackageReference section */
\t\t{PKG_REF} /* XCRemoteSwiftPackageReference "UltralyticsYOLO" */ = {{
\t\t\tisa = XCRemoteSwiftPackageReference;
\t\t\trepositoryURL = "{REPO_URL}";
\t\t\trequirement = {{
\t\t\t\tkind = upToNextMajorVersion;
\t\t\t\tminimumVersion = {MIN_VERSION};
\t\t\t}};
\t\t}};
/* End XCRemoteSwiftPackageReference section */
"""

PROD_SECTION = f"""
/* Begin XCSwiftPackageProductDependency section */
\t\t{PROD_REF} /* UltralyticsYOLO */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_REF} /* XCRemoteSwiftPackageReference "UltralyticsYOLO" */;
\t\t\tproductName = UltralyticsYOLO;
\t\t}};
/* End XCSwiftPackageProductDependency section */
"""

LINK_ENTRY = (
    f"\t\t{LINK_REF} /* UltralyticsYOLO in Frameworks */ = "
    f"{{isa = PBXBuildFile; productRef = {PROD_REF} /* UltralyticsYOLO */; }};\n"
)
FRAMEWORK_PHASE_ENTRY = f"\t\t\t\t{LINK_REF} /* UltralyticsYOLO in Frameworks */,\n"
TARGET_PKGS = (
    "\t\t\tpackageProductDependencies = (\n"
    f"\t\t\t\t{PROD_REF} /* UltralyticsYOLO */,\n"
    "\t\t\t);\n"
)
PROJECT_PKGS = (
    "\t\t\tpackageReferences = (\n"
    f"\t\t\t\t{PKG_REF} /* XCRemoteSwiftPackageReference \"UltralyticsYOLO\" */,\n"
    "\t\t\t);\n"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def remove_script_phase(text: str) -> str:
    """Delete the whole [CP] Check Pods Manifest.lock PBXShellScriptBuildPhase block."""
    start = text.find("\t\t6EC052693C1ABD4570E06A08 /* [CP] Check Pods Manifest.lock */ = {")
    if start < 0:
        return text
    end = text.find("\t\t};\n", start)
    if end < 0:
        fail("unterminated [CP] script phase block")
    return text[:start] + text[end + len("\t\t};\n"):]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    args = parser.parse_args()

    if not PBX.exists():
        fail(f"project not found: {PBX}")

    text = PBX.read_text(encoding="utf-8")

    if REPO_URL in text:
        print(f"already patched ({REPO_URL} present) - nothing to do")
        return 0

    print("Removing CocoaPods / onnxruntime references ...")
    text = remove_script_phase(text)
    for block in REMOVE_BLOCKS:
        if block in text:
            text = text.replace(block, "", 1)
        elif not block.startswith("\t\t6EC05269"):
            print(f"  (note) block not found, skipping: {block.splitlines()[0][:70]}")
    for line in REMOVE_LINES:
        if line in text:
            text = text.replace(line, "", 1)
        else:
            print(f"  (note) line not found, skipping: {line.strip()[:70]}")

    print("Adding UltralyticsYOLO Swift Package dependency ...")
    # 1. build file entry (PBXBuildFile section is currently empty)
    anchor = "/* Begin PBXBuildFile section */\n"
    if anchor not in text:
        fail("PBXBuildFile section marker missing")
    text = text.replace(anchor, anchor + LINK_ENTRY, 1)

    # 2. link in the Frameworks build phase
    anchor = (
        "\t\tLPR00000000000000000000C1 /* Frameworks */ = {\n"
        "\t\t\tisa = PBXFrameworksBuildPhase;\n"
        "\t\t\tbuildActionMask = 2147483647;\n"
        "\t\t\tfiles = (\n"
    )
    if anchor not in text:
        fail("Frameworks build phase anchor missing")
    text = text.replace(anchor, anchor + FRAMEWORK_PHASE_ENTRY, 1)

    # 3. product dependency section (before XCBuildConfiguration)
    anchor = "/* Begin XCBuildConfiguration section */"
    if anchor not in text:
        fail("XCBuildConfiguration section marker missing")
    text = text.replace(anchor, PKG_SECTION.lstrip("\n") + PROD_SECTION.lstrip("\n") + anchor, 1)

    # 4. target packageProductDependencies
    anchor = "\t\t\tproductType = \"com.apple.product-type.application\";\n"
    if anchor not in text:
        fail("target productType anchor missing")
    text = text.replace(anchor, anchor + TARGET_PKGS, 1)

    # 5. project packageReferences
    anchor = "\t\t\tprojectRoot = \"\";\n"
    if anchor not in text:
        fail("projectRoot anchor missing")
    text = text.replace(anchor, anchor + PROJECT_PKGS, 1)

    # ---- validate ----------------------------------------------------------
    problems: list[str] = []
    if text.count("{") != text.count("}"):
        problems.append("unbalanced braces")
    for needle in (REPO_URL, "UltralyticsYOLO", PKG_REF, PROD_REF, LINK_REF):
        if needle not in text:
            problems.append(f"missing after patch: {needle}")
    for leftover in ("onnxruntime", "Pods.xcodeproj", "libPods-LicensePlateOCR.a",
                     "[CP] Check Pods Manifest.lock"):
        if leftover in text:
            problems.append(f"still present: {leftover}")
    if problems:
        for p in problems:
            print(f"  FAIL {p}", file=sys.stderr)
        fail("validation failed - file NOT written")

    if args.check:
        print("--check: validation passed, file not written")
        return 0

    if not BACKUP.exists():
        BACKUP.write_bytes(PBX.read_bytes())
        print(f"backup -> {BACKUP.name}")
    PBX.write_text(text, encoding="utf-8")
    print(f"patched {PBX}")
    print("Next: open the project in Xcode and let it resolve the package")
    print("      (File > Packages > Resolve Package Versions if it does not do so).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
