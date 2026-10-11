/*
 License Plate OCR SDK — Demo App
 
 Abstract:
 App entry point. The recognition engine lives in `LicensePlateSDK` and is reusable;
 this app is just a thin demo that lets a user pick an image or a video, choose the
 `fast` / `accurate` mode, and see the detected German plates.
 */

import SwiftUI

@main
struct LicensePlateOCRApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
