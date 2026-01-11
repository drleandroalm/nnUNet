// Sources/nnUNetPreprocessing/Metal/Shaders/CTNormalization.metal

#include <metal_stdlib>
using namespace metal;

/// CT normalization parameters
struct CTNormParams {
    float mean;
    float std;
    float lowerBound;
    float upperBound;
};

/// CT normalization kernel
/// Clips to percentile bounds and applies z-score normalization
kernel void ct_normalize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant CTNormParams& params [[buffer(2)]],
    constant uint& voxelCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= voxelCount) return;

    float value = input[gid];

    // Clip to percentile bounds
    value = clamp(value, params.lowerBound, params.upperBound);

    // Z-score normalization
    value = (value - params.mean) / max(params.std, 1e-8f);

    output[gid] = value;
}
