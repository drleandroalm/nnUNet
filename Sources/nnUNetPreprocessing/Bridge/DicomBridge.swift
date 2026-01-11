// Sources/nnUNetPreprocessing/Bridge/DicomBridge.swift

import Foundation
import DicomCore
import simd

/// Bridge between DICOM-Decoder and internal VolumeBuffer representation
public struct DicomBridge: Sendable {

    /// Convert DICOM-Decoder output to internal VolumeBuffer with HU conversion
    /// - Parameter volume: DicomSeriesVolume from DICOM-Decoder
    /// - Returns: VolumeBuffer with Float32 data in HU units
    public static func convert(_ volume: DicomSeriesVolume) -> VolumeBuffer {
        let voxelCount = volume.width * volume.height * volume.depth

        // Apply HU conversion: HU = raw * slope + intercept
        let float32Data = volume.voxels.withUnsafeBytes { rawBuffer -> Data in
            var floatArray = [Float](repeating: 0, count: voxelCount)

            let slope = Float(volume.rescaleSlope)
            let intercept = Float(volume.rescaleIntercept)

            if volume.isSignedPixel {
                let int16Ptr = rawBuffer.bindMemory(to: Int16.self)
                for i in 0..<voxelCount {
                    floatArray[i] = Float(int16Ptr[i]) * slope + intercept
                }
            } else {
                let uint16Ptr = rawBuffer.bindMemory(to: UInt16.self)
                for i in 0..<voxelCount {
                    floatArray[i] = Float(uint16Ptr[i]) * slope + intercept
                }
            }

            return floatArray.withUnsafeBytes { Data($0) }
        }

        // DICOM-Decoder uses W×H×D order, we convert to D×H×W for nnUNet
        // Note: The actual data layout remains the same, we just interpret axes differently
        // Full transpose will be handled in the Transpose step based on plans
        return VolumeBuffer(
            data: float32Data,
            shape: (depth: volume.depth, height: volume.height, width: volume.width),
            spacing: SIMD3(volume.spacing.z, volume.spacing.y, volume.spacing.x),
            origin: volume.origin,
            orientation: volume.orientation
        )
    }
}
