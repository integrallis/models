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

private let lastErrorLock = NSLock()
private var lastErrorPointer: UnsafeMutablePointer<CChar>? =
    strdup("Apple Foundation Models bridge has not been initialized")

private func setLastError(_ message: String) {
    lastErrorLock.lock()
    defer { lastErrorLock.unlock() }
    if let lastErrorPointer {
        free(lastErrorPointer)
    }
    lastErrorPointer = strdup(message)
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
            lock.lock()
            result = .success(value)
            lock.unlock()
        } catch {
            lock.lock()
            result = .failure(error)
            lock.unlock()
        }
        semaphore.signal()
    }

    semaphore.wait()
    lock.lock()
    defer { lock.unlock() }
    return try result!.get()
}

@_cdecl("jmodels_afm_available")
public func jmodels_afm_available() -> Int32 {
    let model = SystemLanguageModel.default
    let availability = availabilityMessage(model)
    setLastError(availability.1)
    return availability.0 ? 1 : 0
}

@_cdecl("jmodels_afm_generate")
public func jmodels_afm_generate(
    _ promptPointer: UnsafePointer<CChar>?,
    _ instructionsPointer: UnsafePointer<CChar>?,
    _ maxOutputTokens: Int32
) -> UnsafeMutablePointer<CChar>? {
    guard let promptPointer else {
        setLastError("prompt pointer was null")
        return nil
    }

    let model = SystemLanguageModel.default
    let availability = availabilityMessage(model)
    guard availability.0 else {
        setLastError(availability.1)
        return nil
    }

    let prompt = String(cString: promptPointer)
    let instructions = instructionsPointer.map { String(cString: $0) } ?? ""
    let boundedMaxOutputTokens = max(1, Int(maxOutputTokens))

    do {
        let text = try runBlocking {
            let session =
                instructions.isEmpty
                ? LanguageModelSession(model: model)
                : LanguageModelSession(model: model, instructions: instructions)
            let options = GenerationOptions(maximumResponseTokens: boundedMaxOutputTokens)
            let response = try await session.respond(to: prompt, options: options)
            return response.content
        }
        setLastError("ok")
        return strdup(text)
    } catch {
        setLastError(String(describing: error))
        return nil
    }
}

@_cdecl("jmodels_afm_last_error")
public func jmodels_afm_last_error() -> UnsafeMutablePointer<CChar>? {
    lastErrorLock.lock()
    defer { lastErrorLock.unlock() }
    guard let lastErrorPointer else {
        return nil
    }
    return strdup(lastErrorPointer)
}

@_cdecl("jmodels_afm_free")
public func jmodels_afm_free(_ pointer: UnsafeMutablePointer<CChar>?) {
    if let pointer {
        free(pointer)
    }
}
