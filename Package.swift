// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "nnUNetPreprocessing",
    platforms: [
        .iOS(.v26),
        .macOS(.v26)
    ],
    products: [
        .library(
            name: "nnUNetPreprocessing",
            targets: ["nnUNetPreprocessing"]
        )
    ],
    dependencies: [
        .package(path: "../DICOM-Decoder"),
        .package(path: "../MTK"),
    ],
    targets: [
        .target(
            name: "nnUNetPreprocessing",
            dependencies: [
                .product(name: "DicomCore", package: "DICOM-Decoder"),
                .product(name: "MTKCore", package: "MTK"),
            ],
            exclude: [
                "CLAUDE.md",
                "Bridge/CLAUDE.md",
                "CPU/CLAUDE.md",
                "Metal/CLAUDE.md",
                "Metal/Shaders/CLAUDE.md",
                "Models/CLAUDE.md",
            ],
            resources: [
                .process("Metal/Shaders")
            ]
        ),
        .testTarget(
            name: "nnUNetPreprocessingTests",
            dependencies: ["nnUNetPreprocessing"],
            exclude: [
                "CLAUDE.md",
                "Helpers/CLAUDE.md",
                "Fixtures/CLAUDE.md",
            ],
            resources: [
                .process("Fixtures")
            ]
        )
    ]
)
