# Original NB model converted to Core ML

This branch uses the **same plate detector and FP32 parameters** as `main` at `9ec6f24bd75b5486fea4df176ae98e1108b75d70`. It does not use a replacement model or the old COCO car detector.

The runtime model is `LicensePlateOCR/LicensePlateSDK/Resources/license_plate_yolo.mlmodel`. `Models/source/yolo.nb` is retained solely as an offline conversion and verification input, outside the app and Swift Package resource directories. The runtime has no Paddle Lite library, native bridge, or `.nb` model resource.

## Conversion

The NB file has version 2, optimizer `v2.14-rc`, a FlatBuffer inference graph, and 129 FP32 parameter tensors. Its operations are mapped to built-in Core ML neural-network layers. The conversion preserves every original parameter byte, uses static tensor shapes, and replaces reused Lite blob names with unique Core ML graph names. SiLU is represented by sigmoid and multiplication. No custom Paddle operators, retraining, replacement weights, or quantization are introduced.

The output format is a Core ML `.mlmodel` neural-network specification, version 5, rather than an ML Program package. Both are Core ML formats. A native Core ML specification can be written on Windows using the protobuf schemas; it still requires Apple's compiler and runtime for device validation. Core ML compute units are set to `.all`, allowing eligible CPU, GPU and Neural Engine execution. This setting does not guarantee that every layer uses the Neural Engine.

The model accepts a 640 × 640 BGR image. It incorporates the baseline preprocessing: `(pixel / 255 - mean) * inverse_std`, with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The SDK supplies the original centered 114-gray letterbox padding. Predictions are `[1, 5, 8400]`: center x/y, width/height in input pixels, and plate probabilities. The SDK reverses letterboxing and applies the baseline confidence and NMS thresholds, both 0.45.

Source provenance is in [nb-source-provenance.json](nb-source-provenance.json). Export metadata, SHA-256 hashes and all parameter-to-layer mappings are in [nb-coreml-provenance.json](nb-coreml-provenance.json).

## Reproduce

From `OCR/`, with Python 3.12 and an isolated environment:

```sh
python -m pip install numpy protobuf coremltools
python Scripts/convert_nb_to_coreml.py \
  --source Models/source/yolo.nb \
  --output LicensePlateOCR/LicensePlateSDK/Resources/license_plate_yolo.mlmodel \
  --report Models/nb-coreml-provenance.json
```

Only `coremltools`' generated protobuf definitions are needed to write the model. On platforms without a supported Core ML Tools installation, `--schema-dir` can point to the `coremltools/proto` directory extracted from an official wheel. The conversion does not invoke macOS native libraries.

For CPU graph verification:

```sh
python -m pip install torch onnx onnxruntime
python Scripts/validate_nb_coreml.py \
  --source Models/source/yolo.nb \
  --coreml LicensePlateOCR/LicensePlateSDK/Resources/license_plate_yolo.mlmodel \
  --provenance Models/nb-coreml-provenance.json \
  --report Models/nb-coreml-validation.json
```

## Completed validation and limits

All 129 recovered FP32 parameter tensors match the NB source bit-for-bit. The serialized Core ML graph passes ONNX graph validation when mapped independently to corresponding operations. A PyTorch evaluator of the original NB graph and an ONNX Runtime evaluator of the serialized Core ML graph agree on black, 114-gray and seeded-random inputs. The maximum observed coordinate difference is approximately 0.00165 model-input pixels; the maximum score difference is approximately 6.52e-7. Detailed evidence is in [nb-coreml-validation.json](nb-coreml-validation.json).

These are **synthetic graph-equivalence checks**, not runs of the actual Paddle Lite or Apple Core ML runtime. Xcode compilation, Apple runtime numerical parity, real plate accuracy, Neural Engine placement, latency and energy remain unverified in the Windows workspace. The Swift regression tests cover raw YOLO output layouts/storage types, clipping/NMS, the real model's compilation/loading, and the existing frame pipeline; they require macOS/Xcode to execute.

For the device comparison, use identical recorded frames, OCR mode, frame rate and thresholds on both branches. Measure detection separately from preprocessing and OCR; then compare end-to-end latency, memory and sustained thermal behavior. Verify predictions on representative plate images and inspect Xcode's Core ML performance report before claiming faster or more efficient detection.

Run `bash Scripts/validate_apple_build.sh` from `OCR/` on macOS to build device and simulator destinations and run the package tests. Shared OCR, sampling and camera fixes have also been applied to the NB worktree at `E:/Git/ITLogica/PlateOCR-de-nb-review`. Its Paddle runtime has been materialized into native arm64 objects; CPU performance or rounding may differ from the historical bitcode baseline even though the plate model's weights are unchanged. Neither working tree has been committed by this task.

## Format references

- [Paddle Lite v2.14-rc model reader](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.14-rc/lite/model_parser/model_parser.cc)
- [Paddle FlatBuffer graph schema](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.14-rc/lite/model_parser/flatbuffers/framework.fbs)
- [Paddle parameter schema](https://github.com/PaddlePaddle/Paddle-Lite/blob/v2.14-rc/lite/model_parser/flatbuffers/param.fbs)
- [Apple Core ML neural-network specification](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html)
- [Apple Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)
