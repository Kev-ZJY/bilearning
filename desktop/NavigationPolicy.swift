import Foundation

enum WorkbenchNavigationPolicy {
    static func allows(url: URL, baseURL: URL?, isMainFrame: Bool) -> Bool {
        // Only the official player may navigate inside an iframe. Remote pages
        // still cannot replace the app or use its main-frame-only native bridge.
        if !isMainFrame && url.scheme == "https" && url.host == "player.bilibili.com"
            && url.path == "/player.html" && (url.port == nil || url.port == 443)
            && url.user == nil && url.password == nil {
            return true
        }
        return url.scheme == "about" || (url.scheme == baseURL?.scheme
            && url.host == baseURL?.host && url.port == baseURL?.port)
    }
}
