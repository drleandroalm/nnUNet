// Sources/nnUNetPreprocessing/Metal/PreprocessingShaderLibraryLoader.swift

import Foundation
import Metal

enum PreprocessingShaderLibraryLoader {
    static func makeDefaultLibrary(
        on device: MTLDevice,
        diagnostics: (String) -> Void = { _ in }
    ) -> MTLLibrary? {
        if let library = loadBundledMetallib(on: device, diagnostics: diagnostics) {
            return library
        }

        if #available(iOS 14, macOS 11, *) {
            if let bundled = try? device.makeDefaultLibrary(bundle: .module) {
                diagnostics("[nnUNetPreprocessing] Loaded compiled shaders via Bundle.module default library")
                return bundled
            }
        }

        if let main = device.makeDefaultLibrary() {
            diagnostics("[nnUNetPreprocessing] Loaded main bundle default library")
            return main
        }

#if DEBUG
        if let runtimeLib = runtimeLibrary(on: device, diagnostics: diagnostics) {
            return runtimeLib
        }
#endif

        diagnostics("[nnUNetPreprocessing] Unable to load Metal library")
        return nil
    }
}

private extension PreprocessingShaderLibraryLoader {
    static func loadBundledMetallib(
        on device: MTLDevice,
        diagnostics: (String) -> Void
    ) -> MTLLibrary? {
        let urls = resourceURLs(withExtension: "metallib")
        guard !urls.isEmpty else { return nil }

        let preferred = urls.first(where: { $0.lastPathComponent == "default.metallib" })
        let url = preferred ?? urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }).first!

        do {
            let library = try device.makeLibrary(URL: url)
            diagnostics("[nnUNetPreprocessing] Loaded precompiled metallib: \(url.lastPathComponent)")
            return library
        } catch {
            diagnostics("[nnUNetPreprocessing] Failed to load metallib (\(url.lastPathComponent)): \(error)")
            return nil
        }
    }

    static func runtimeLibrary(
        on device: MTLDevice,
        diagnostics: (String) -> Void
    ) -> MTLLibrary? {
        guard let source = concatenatedShaderSources() else {
            diagnostics("[nnUNetPreprocessing] No shader source files available for runtime compilation")
            return nil
        }

        let options = MTLCompileOptions()
        if #available(iOS 16, macOS 13, *) {
            options.languageVersion = .version3_0
        }

        do {
            let library = try device.makeLibrary(source: source, options: options)
            diagnostics("[nnUNetPreprocessing] Compiled shaders at runtime as a fallback")
            return library
        } catch {
            diagnostics("[nnUNetPreprocessing] Runtime shader compilation failed: \(error)")
            return nil
        }
    }

    static func concatenatedShaderSources() -> String? {
        let urls = resourceURLs(withExtension: "metal")
        guard !urls.isEmpty else { return nil }

        let sources = urls
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            .compactMap { try? String(contentsOf: $0, encoding: .utf8) }
        guard !sources.isEmpty else { return nil }

        return sources.joined(separator: "\n\n")
    }

    static func resourceURLs(withExtension fileExtension: String) -> [URL] {
        var urls: [URL] = []
        if let direct = Bundle.module.urls(forResourcesWithExtension: fileExtension, subdirectory: nil) {
            urls.append(contentsOf: direct)
        }
        if let shadersDir = Bundle.module.urls(forResourcesWithExtension: fileExtension, subdirectory: "Metal/Shaders") {
            urls.append(contentsOf: shadersDir)
        }

        if urls.isEmpty, let resourceURL = Bundle.module.resourceURL {
            let fileManager = FileManager.default
            if let enumerator = fileManager.enumerator(at: resourceURL, includingPropertiesForKeys: nil) {
                for case let url as URL in enumerator {
                    if url.pathExtension == fileExtension {
                        urls.append(url)
                    }
                }
            }
        }

        var seen = Set<String>()
        return urls
            .filter { seen.insert($0.path).inserted }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
    }
}
