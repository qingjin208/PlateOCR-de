# NB and Core ML branch code review

Reviewed `main` at `9ec6f24bd75b5486fea4df176ae98e1108b75d70` using `git show`, and the uncommitted working tree on `feature/coreml-plate-detection`. After the user requested fixes, all shared corrections were ported to the `main` NB worktree at `E:/Git/ITLogica/PlateOCR-de-nb-review`. Both working trees are patched; branch references have not been advanced and no changes have been committed or pushed.

## Confirmed findings

| Priority | Scope | Finding and trigger | Resolution |
| --- | --- | --- | --- |
| High | Shared OCR post-processing | If Vision returns `HH` and `AB1234` separately, `AB1234` canonicalizes to `A-B 1234`. The old confidence-only choice can prefer that fragment over the complete `HH-AB 1234`, especially when `AB1234` has higher confidence. See `main:OCR/LicensePlateOCR/LicensePlateSDK/PlatePostProcessor.swift:72`. | The current branch prefers a valid complete assembly and includes a regression test with unequal fragment confidence. The same fix is applied to the NB worktree. |
| Medium | Shared OCR ordering | The old comparator switched between horizontal and vertical order using a pairwise height threshold. That can violate transitivity when three observations have different row heights, causing inconsistent assembly order. See the same main file at line 37. | The current branch groups observations into rows first and then sorts each row horizontally. The same fix is applied to the NB worktree. |
| Medium | Shared file-video processing | Reaching `maxSamples` merely skipped analysis while still decoding all remaining frames. For `maxSamples = 1`, a long video was decoded to its end after processing the first frame. See `main:OCR/LicensePlateOCR/LicensePlateSDK/LicensePlateRecognizer.swift:191`. | The current branch exits the decode loop at the sample limit, checks finite duration and scopes synchronous sample/render work to autorelease pools. The same fix is applied to the NB worktree. |
| Medium | Shared camera preview | Portrait rotation was configured only during view creation, when the camera input connection could still be absent. `updateUIView` did not retry it. See `main:OCR/LicensePlateOCR/Camera/CameraPreview.swift:26`. | The current branch retries portrait rotation during view updates and layout. Alignment must still be checked on a device. The same fix is applied to the NB worktree. |
| Medium | Shared live callbacks | The early detection callback used a session generation token but no per-frame identifier. A delayed callback could arrive after the next frame had started, pass the `waitingForOCR` check, and draw stale boxes. See `main:OCR/LicensePlateOCR/Camera/CameraController.swift:135`. | The current branch adds a per-analysis identifier and rejects callbacks from earlier frames. It also avoids starting capture after its setup task is cancelled. The same fixes are applied to the NB worktree. |

Updated code: [PlatePostProcessor.swift](../LicensePlateOCR/LicensePlateSDK/PlatePostProcessor.swift), [LicensePlateRecognizer.swift](../LicensePlateOCR/LicensePlateSDK/LicensePlateRecognizer.swift), [CameraController.swift](../LicensePlateOCR/Camera/CameraController.swift), and [CameraPreview.swift](../LicensePlateOCR/Camera/CameraPreview.swift).

## Model presence and integrity

`LicensePlateOCR/LicensePlateSDK/Resources/license_plate_yolo.mlmodel` exists in the current working tree. Its size is **12,177,860 bytes** and SHA-256 is `90037c96f83346f6f395c3be9623ba173c05eb15e3fc7b6454853c5b9643b5bb`, matching the conversion provenance. It is an untracked file and is not yet part of a branch commit; both branch tips currently remain at the NB baseline commit.

The protobuf parses as a Core ML neural network with specification version 5, 430 built-in layers, a 640 × 640 BGR image input and `[1, 5, 8400]` output. Graph connectivity, unique output names, positive static shapes and parameter hashes pass inspection. All 129 parameter tensors match the original NB model bit-for-bit. CPU graph-equivalence checks pass again for black, 114-gray and seeded-random inputs.

The current Xcode target and Swift Package contain no Paddle runtime or native bridge references. The original NB input is retained outside app resources in `Models/source/yolo.nb`.

## Remaining verification boundaries

- **Core ML branch:** The model has not been compiled or executed by Apple's Core ML runtime. Protobuf and ONNX graph checks cannot establish acceptance by `coremlc`, hardware placement, real detection accuracy or energy benefits. The added Swift tests load the actual model and compare Apple CPU predictions against original-graph fixtures; these remain unexecuted on Windows.
- **NB main worktree:** The 528 old bitcode modules have been materialized as native arm64 iOS 18 Mach-O objects. The upstream SIMD vector-clobber declarations were corrected without changing assembly instructions. Required API/registration symbols and all object targets were inspected. Device-only autolinking was verified using compiled device/simulator/Catalyst fixtures. Full Apple SDK linking and NB inference still require Apple-host/device validation.
- **Both implementations:** Separated EU-strip markers now yield the remaining complete plate; actual Dresden `D` prefixes are preserved, arbitrary leading-character removal is eliminated, and validation rejects trailing newlines. Missing separators can still be inherently ambiguous; representative plate images and two-line plates need accuracy checks.
- The Swift regression test added for complete-plate assembly has not run here because the Windows environment lacks Apple frameworks. The camera/video fixes are code-reviewed but also require Apple-platform execution.

## Checks executed

- Parsed the converted model and checked its file hash, graph connectivity and static shape metadata.
- Reran original-weight verification and all three independent CPU graph-equivalence cases successfully.
- Inspected both runtime loaders, pixel preprocessing, tensor decoding, camera flow, OCR post-processing and package/project configuration.
- Ran `git diff --check` successfully.

No claim is made that either branch is fully validated on an iPhone, or that the converted model is faster or more energy-efficient.

## Follow-up repair validation

- Shared source files and OCR regression tests are synchronized across the Core ML and NB worktrees. Their default detector implementations remain Core ML and Paddle respectively.
- The NB C++ decoder tests were compiled with Zig bundled Clang and executed on Windows successfully.
- Native runtime materialization compiled all 528 modules with inline-assembly warnings treated as errors. All archive members are arm64 iOS 18 objects.
- The actual native wrapper guard/autolink directive produced the library request only in the device fixture; simulator/Catalyst fixtures did not request the Paddle device library.
- Both worktrees include `Scripts/validate_apple_build.sh` for device/simulator builds and Swift tests on a Mac. These Apple-host checks remain unexecuted in this environment.
