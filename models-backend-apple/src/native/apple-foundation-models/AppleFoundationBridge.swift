/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import Darwin
import Dispatch
import Foundation
import FoundationModels

private func makeResult(
    status: Int32,
    value: Int32 = 0,
    text: String
) -> UnsafeMutablePointer<jmodels_afm_result>? {
    guard
        let result =
            malloc(MemoryLayout<jmodels_afm_result>.stride)?
                .assumingMemoryBound(to: jmodels_afm_result.self)
    else {
        return nil
    }
    result.initialize(to: jmodels_afm_result())
    result.pointee.status = status
    result.pointee.value = value

    let bytes = Array(text.utf8)
    if !bytes.isEmpty {
        guard let data = malloc(bytes.count)?.assumingMemoryBound(to: UInt8.self) else {
            result.deinitialize(count: 1)
            free(result)
            return nil
        }
        _ = bytes.withUnsafeBytes { source in
            memcpy(data, source.baseAddress!, bytes.count)
        }
        result.pointee.data = data
        result.pointee.length = bytes.count
    }
    return result
}

private func decode(
    _ pointer: UnsafePointer<UInt8>?,
    length: UInt,
    name: String
) throws -> String {
    guard let count = Int(exactly: length) else {
        throw BridgeInputError.invalidLength(name)
    }
    if count == 0 {
        return ""
    }
    guard let pointer else {
        throw BridgeInputError.nullPayload(name)
    }
    let bytes = UnsafeBufferPointer(start: pointer, count: count)
    guard let text = String(bytes: bytes, encoding: .utf8) else {
        throw BridgeInputError.invalidUtf8(name)
    }
    return text
}

private enum BridgeInputError: Error, CustomStringConvertible {
    case invalidLength(String)
    case nullPayload(String)
    case invalidUtf8(String)

    var description: String {
        switch self {
        case .invalidLength(let name):
            return "\(name) length cannot be represented"
        case .nullPayload(let name):
            return "\(name) payload was null"
        case .invalidUtf8(let name):
            return "\(name) payload was not valid UTF-8"
        }
    }
}

private func availabilityMessage(_ model: SystemLanguageModel) -> (Bool, String) {
    switch model.availability {
    case .available:
        return (true, "Apple Foundation Models available")
    case .unavailable(let reason):
        switch reason {
        case .appleIntelligenceNotEnabled:
            return (false, "Apple Intelligence is not enabled")
        case .deviceNotEligible:
            return (false, "device is not eligible for Apple Intelligence")
        case .modelNotReady:
            return (false, "Apple Intelligence model is not ready")
        @unknown default:
            return (false, "Apple Foundation Models unavailable for an unknown reason")
        }
    @unknown default:
        return (false, "Apple Foundation Models unavailable for an unknown reason")
    }
}

private func runBlocking<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let lock = NSLock()
    var result: Result<T, Error>?

    Task {
        do {
            let value = try await operation()
            lock.withLock {
                result = .success(value)
            }
        } catch {
            lock.withLock {
                result = .failure(error)
            }
        }
        semaphore.signal()
    }

    semaphore.wait()
    return try lock.withLock {
        try result!.get()
    }
}

@_cdecl("jmodels_afm_available")
public func jmodelsAppleFoundationAvailable() -> UnsafeMutablePointer<jmodels_afm_result>? {
    let availability = availabilityMessage(SystemLanguageModel.default)
    return makeResult(status: 0, value: availability.0 ? 1 : 0, text: availability.1)
}

@_cdecl("jmodels_afm_generate")
public func jmodelsAppleFoundationGenerate(
    _ promptPointer: UnsafePointer<UInt8>?,
    _ promptLength: UInt,
    _ instructionsPointer: UnsafePointer<UInt8>?,
    _ instructionsLength: UInt,
    _ maxOutputTokens: Int32
) -> UnsafeMutablePointer<jmodels_afm_result>? {
    do {
        let prompt = try decode(promptPointer, length: promptLength, name: "prompt")
        let instructions =
            try decode(instructionsPointer, length: instructionsLength, name: "instructions")
        let model = SystemLanguageModel.default
        let availability = availabilityMessage(model)
        guard availability.0 else {
            return makeResult(status: 1, text: availability.1)
        }

        let boundedMaxOutputTokens = max(1, Int(maxOutputTokens))
        let text = try runBlocking {
            let session =
                instructions.isEmpty
                ? LanguageModelSession(model: model)
                : LanguageModelSession(model: model, instructions: instructions)
            let options = GenerationOptions(maximumResponseTokens: boundedMaxOutputTokens)
            let response = try await session.respond(to: prompt, options: options)
            return response.content
        }
        return makeResult(status: 0, text: text)
    } catch {
        return makeResult(status: 1, text: String(describing: error))
    }
}

@_cdecl("jmodels_afm_result_free")
public func jmodelsAppleFoundationResultFree(
    _ result: UnsafeMutablePointer<jmodels_afm_result>?
) {
    guard let result else {
        return
    }
    free(result.pointee.data)
    result.deinitialize(count: 1)
    free(result)
}
