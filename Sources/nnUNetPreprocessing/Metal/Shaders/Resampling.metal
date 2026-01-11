// Sources/nnUNetPreprocessing/Metal/Shaders/Resampling.metal

#include <metal_stdlib>
using namespace metal;

/// Resampling parameters for volume transformation
struct ResampleParams {
    // Source volume dimensions
    uint srcDepth;
    uint srcHeight;
    uint srcWidth;

    // Target volume dimensions
    uint dstDepth;
    uint dstHeight;
    uint dstWidth;

    // Scale factors for coordinate mapping
    float scaleZ;
    float scaleY;
    float scaleX;

    // Interpolation order for Z axis (0 = nearest, 1 = linear)
    uint orderZ;
};

/// Cubic B-spline weight function matching skimage order=3
inline float cubicWeight(float t) {
    float at = abs(t);
    if (at < 1.0f) {
        return (1.5f * at - 2.5f) * at * at + 1.0f;
    } else if (at < 2.0f) {
        return ((-0.5f * at + 2.5f) * at - 4.0f) * at + 2.0f;
    }
    return 0.0f;
}

/// Edge-clamped coordinate (mode='edge' in skimage)
inline int edgeClamp(int coord, int size) {
    return clamp(coord, 0, size - 1);
}

/// 3D cubic B-spline interpolation with edge padding
/// Uses 4×4×4 = 64 neighbor sampling
kernel void resample_cubic_3d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ResampleParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= params.dstWidth || gid.y >= params.dstHeight || gid.z >= params.dstDepth) {
        return;
    }

    // Map to source coordinates
    float sx = float(gid.x) * params.scaleX;
    float sy = float(gid.y) * params.scaleY;
    float sz = float(gid.z) * params.scaleZ;

    // Get integer and fractional parts
    int ix = int(floor(sx));
    int iy = int(floor(sy));
    int iz = int(floor(sz));

    float fx = sx - float(ix);
    float fy = sy - float(iy);
    float fz = sz - float(iz);

    // Accumulate weighted sum over 4×4×4 neighborhood
    float result = 0.0f;

    for (int dz = -1; dz <= 2; dz++) {
        float wz = cubicWeight(fz - float(dz));
        int cz = edgeClamp(iz + dz, int(params.srcDepth));

        for (int dy = -1; dy <= 2; dy++) {
            float wy = cubicWeight(fy - float(dy));
            int cy = edgeClamp(iy + dy, int(params.srcHeight));

            for (int dx = -1; dx <= 2; dx++) {
                float wx = cubicWeight(fx - float(dx));
                int cx = edgeClamp(ix + dx, int(params.srcWidth));

                uint srcIdx = uint(cz) * params.srcHeight * params.srcWidth +
                              uint(cy) * params.srcWidth + uint(cx);
                result += input[srcIdx] * wz * wy * wx;
            }
        }
    }

    // Write output
    uint dstIdx = gid.z * params.dstHeight * params.dstWidth +
                  gid.y * params.dstWidth + gid.x;
    output[dstIdx] = result;
}

/// 2D cubic interpolation for separate-Z mode (first pass: in-plane resampling)
/// Resamples X-Y for each source Z slice
kernel void resample_separate_z_xy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ResampleParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // In this pass: output is [srcDepth × dstHeight × dstWidth]
    // gid.z indexes source slices, gid.y/x index target XY
    if (gid.x >= params.dstWidth || gid.y >= params.dstHeight || gid.z >= params.srcDepth) {
        return;
    }

    // Map to source XY coordinates
    float sx = float(gid.x) * params.scaleX;
    float sy = float(gid.y) * params.scaleY;

    int ix = int(floor(sx));
    int iy = int(floor(sy));

    float fx = sx - float(ix);
    float fy = sy - float(iy);

    // Slice offset in source
    uint sliceOffset = gid.z * params.srcHeight * params.srcWidth;

    // 2D cubic interpolation (4×4 neighborhood)
    float result = 0.0f;

    for (int dy = -1; dy <= 2; dy++) {
        float wy = cubicWeight(fy - float(dy));
        int cy = edgeClamp(iy + dy, int(params.srcHeight));

        for (int dx = -1; dx <= 2; dx++) {
            float wx = cubicWeight(fx - float(dx));
            int cx = edgeClamp(ix + dx, int(params.srcWidth));

            uint srcIdx = sliceOffset + uint(cy) * params.srcWidth + uint(cx);
            result += input[srcIdx] * wy * wx;
        }
    }

    // Write to intermediate buffer [srcDepth × dstHeight × dstWidth]
    uint dstIdx = gid.z * params.dstHeight * params.dstWidth +
                  gid.y * params.dstWidth + gid.x;
    output[dstIdx] = result;
}

/// Z-axis interpolation for separate-Z mode (second pass)
/// Resamples through-plane with nearest-neighbor (orderZ=0) or linear (orderZ=1)
kernel void resample_separate_z_z(
    device const float* input [[buffer(0)]],   // Intermediate: [srcDepth × dstHeight × dstWidth]
    device float* output [[buffer(1)]],        // Final: [dstDepth × dstHeight × dstWidth]
    constant ResampleParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dstWidth || gid.y >= params.dstHeight || gid.z >= params.dstDepth) {
        return;
    }

    // Map to source Z coordinate
    float sz = float(gid.z) * params.scaleZ;

    float value;
    uint intermediateSliceSize = params.dstHeight * params.dstWidth;
    uint xyOffset = gid.y * params.dstWidth + gid.x;

    if (params.orderZ == 0) {
        // Nearest-neighbor
        uint nearestZ = uint(clamp(int(round(sz)), 0, int(params.srcDepth) - 1));
        uint srcIdx = nearestZ * intermediateSliceSize + xyOffset;
        value = input[srcIdx];
    } else {
        // Linear interpolation
        int z0 = int(floor(sz));
        int z1 = min(z0 + 1, int(params.srcDepth) - 1);
        float t = sz - float(z0);

        uint idx0 = uint(z0) * intermediateSliceSize + xyOffset;
        uint idx1 = uint(z1) * intermediateSliceSize + xyOffset;

        value = input[idx0] * (1.0f - t) + input[idx1] * t;
    }

    // Write output
    uint dstIdx = gid.z * params.dstHeight * params.dstWidth +
                  gid.y * params.dstWidth + gid.x;
    output[dstIdx] = value;
}
