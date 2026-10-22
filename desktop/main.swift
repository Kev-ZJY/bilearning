import AppKit
import WebKit
import Security

// Small command mode used by Python; secret data travels only over stdin/stdout.
if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--keychain" {
    let action = CommandLine.arguments[2]
    let service = (Bundle.main.bundleIdentifier ?? "org.video-workbench.desktop") + ".llm"
    let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service, kSecAttrAccount as String: "user-model"]
    var code: OSStatus = errSecSuccess
    var value = ""
    switch action {
    case "read":
        var q = query; q[kSecReturnData as String] = true; q[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        code = SecItemCopyMatching(q as CFDictionary, &result)
        if let data = result as? Data { value = String(data: data, encoding: .utf8) ?? "" }
        if code == errSecItemNotFound { code = errSecSuccess }
    case "write":
        let data = FileHandle.standardInput.readDataToEndOfFile()
        let input = (try? JSONSerialization.jsonObject(with: data)) as? [String: String]
        guard let text = input?["value"] else { exit(2) }
        let attrs: [String: Any] = [kSecValueData as String: Data(text.utf8)]
        code = SecItemUpdate(query as CFDictionary, attrs as CFDictionary)
        if code == errSecItemNotFound {
            var q = query; q[kSecValueData as String] = Data(text.utf8)
            q[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            code = SecItemAdd(q as CFDictionary, nil)
        }
    case "delete":
        code = SecItemDelete(query as CFDictionary)
        if code == errSecItemNotFound { code = errSecSuccess }
    default: exit(2)
    }
    if code != errSecSuccess { exit(3) }
    let output = try! JSONSerialization.data(withJSONObject: ["value": value])
    FileHandle.standardOutput.write(output)
    exit(0)
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKScriptMessageHandlerWithReply {
    var window: NSWindow!
    var web: WKWebView!
    var process: Process?
    var baseURL: URL?
    let token = UUID().uuidString + UUID().uuidString
    var terminating = false
    var ready = false
    var showingError = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let root = NSMenuItem(); menu.addItem(root)
        let appMenu = NSMenu(); root.submenu = appMenu
        appMenu.addItem(withTitle: "退出视频学习工作台", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let edit = NSMenuItem(title: "编辑", action: nil, keyEquivalent: ""); menu.addItem(edit)
        let editMenu = NSMenu(title: "编辑"); edit.submenu = editMenu
        for (title, action, key) in [("撤销", "undo:", "z"), ("剪切", "cut:", "x"), ("复制", "copy:", "c"), ("粘贴", "paste:", "v"), ("全选", "selectAll:", "a")] {
            editMenu.addItem(withTitle: title, action: Selector(action), keyEquivalent: key)
        }
        NSApp.mainMenu = menu
        window = NSWindow(contentRect: NSRect(x: 0,y: 0,width: 1320,height: 850), styleMask: [.titled,.closable,.miniaturizable,.resizable], backing: .buffered, defer: false)
        window.title = "视频学习工作台"
        window.minSize = NSSize(width: 820,height: 580)
        window.setFrameAutosaveName("WorkbenchMain")
        let config = WKWebViewConfiguration()
        config.userContentController.addScriptMessageHandler(self, contentWorld: .page, name: "desktop")
        web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = self
        window.contentView = web
        window.center(); window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        web.loadHTMLString("<body style='font:16px -apple-system;padding:48px;color:#424857'><h2>正在打开学习工作台</h2><p>启动本地处理服务…</p></body>",baseURL:nil)
        startBackend()
    }

    func showError(_ message: String) {
        if showingError || terminating {return}; showingError=true
        let alert = NSAlert(); alert.messageText = "工作台未能启动"; alert.informativeText = message
        alert.addButton(withTitle: "退出")
        alert.beginSheetModal(for: window) { _ in NSApp.terminate(nil) }
    }

    func startBackend() {
        do {
            let configURL = Bundle.main.resourceURL!.appendingPathComponent("development.json")
            let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as! [String: String]
            guard let python = config["python"], let root = config["root"] else { throw NSError(domain:"config",code:1) }
            let support = config["data"] ?? FileManager.default.urls(for: .applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("VideoWorkbench").path
            try FileManager.default.createDirectory(atPath: support,withIntermediateDirectories:true)
            let proc = Process(); proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m","app.desktop_runner"]
            proc.currentDirectoryURL = URL(fileURLWithPath:root).appendingPathComponent("backend")
            // Explicit allowlist: never inherit LLM_* or DB_URL from developer shell.
            var env: [String:String] = [:]
            for key in ["HOME","USER","TMPDIR","LANG","LC_ALL","SSL_CERT_FILE","REQUESTS_CA_BUNDLE"] {
                if let value=ProcessInfo.processInfo.environment[key] { env[key]=value }
            }
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            env["PYTHONPATH"] = root + "/backend"
            env["STORAGE_ROOT"] = support
            env["DB_URL"] = "sqlite:///" + support + "/workbench.db"
            env["FRONTEND_DIST"] = root + "/frontend/dist"
            env["WORKBENCH_DESKTOP_TOKEN"] = token
            env["WORKBENCH_KEYCHAIN_HELPER"] = Bundle.main.executablePath!
            proc.environment = env
            let output = Pipe(); proc.standardOutput = output
            proc.standardError = FileHandle.nullDevice // Do not retain API payloads or credentials in app logs.
            process = proc
            proc.terminationHandler = { [weak self] _ in
                DispatchQueue.main.async { if let self=self, !self.terminating { self.showError("本地服务已退出。请检查 Python 依赖，重新打开应用；已有项目文件会保留。") } }
            }
            try proc.run()
            DispatchQueue.global().async { [weak self] in
                var bytes = Data()
                while let chunk = try? output.fileHandleForReading.read(upToCount:1), !chunk.isEmpty {
                    if chunk[0] == 10 { break }; bytes.append(chunk)
                    if bytes.count > 8 { break }
                }
                let port = Int(String(data:bytes,encoding:.utf8) ?? "")
                DispatchQueue.main.async {
                    guard let self=self else { return }
                    guard let port=port else { self.showError("无法启动本地服务，请重新构建开发应用。"); return }
                    self.baseURL=URL(string:"http://127.0.0.1:\(port)")!
                    self.waitReady(remaining:120)
                }
            }
        } catch { showError("开发配置或 Python 环境不可用，请重新运行桌面构建命令。") }
    }

    func waitReady(remaining:Int) {
        guard !terminating, let base=baseURL else { return }
        var request=URLRequest(url:base.appendingPathComponent("api/health"))
        request.setValue(token,forHTTPHeaderField:"X-Workbench-Native")
        request.timeoutInterval=1
        URLSession.shared.dataTask(with:request) { [weak self] _,response,_ in
            DispatchQueue.main.async {
                guard let self=self, !self.terminating else {return}
                if (response as? HTTPURLResponse)?.statusCode == 200 {
                    let cookie=HTTPCookie(properties:[.domain:"127.0.0.1",.path:"/",.name:"workbench_session",.value:self.token,HTTPCookiePropertyKey("HttpOnly"):"TRUE",HTTPCookiePropertyKey("SameSite"):"Strict"])!
                    self.web.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) {
                        self.ready=true; self.web.load(URLRequest(url:base))
                    }
                } else if remaining>0 {
                    DispatchQueue.main.asyncAfter(deadline:.now()+0.25) {self.waitReady(remaining:remaining-1)}
                } else {self.showError("本地服务未及时就绪。请检查依赖后重新启动。")}
            }
        }.resume()
    }

    func nativeRequest(_ path:String, body:Any?=nil, reply:@escaping(Any?,String?)->Void) {
        guard let base=baseURL else {reply(nil,"服务未就绪");return}
        var req=URLRequest(url:URL(string:path,relativeTo:base)!.absoluteURL)
        req.setValue(token,forHTTPHeaderField:"X-Workbench-Native")
        if let body=body {req.httpMethod="POST";req.httpBody=try? JSONSerialization.data(withJSONObject:body);req.setValue("application/json",forHTTPHeaderField:"Content-Type")}
        URLSession.shared.dataTask(with:req) {data,response,error in
            let value=data.flatMap {try? JSONSerialization.jsonObject(with:$0)}
            DispatchQueue.main.async {
                if let status=(response as? HTTPURLResponse)?.statusCode, (200..<300).contains(status), let value=value {reply(value,nil)}
                else {reply(nil,(value as? [String:Any])?["detail"] as? String ?? "本地请求失败，请重试")}
            }
        }.resume()
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage, replyHandler: @escaping (Any?, String?) -> Void) {
        guard ready, message.frameInfo.isMainFrame, let url=message.frameInfo.request.url, let base=baseURL, url.host==base.host, url.port==base.port, url.scheme=="http",
              let body=message.body as? [String:Any], let action=body["action"] as? String else {replyHandler(nil,"不允许的页面请求");return}
        if action=="directory" || action=="media" {
            let panel=NSOpenPanel();panel.canChooseDirectories=action=="directory";panel.canChooseFiles=action=="media";panel.allowsMultipleSelection=false;panel.canCreateDirectories=action=="directory"
            panel.prompt=action=="directory" ? "选择项目目录" : "导入媒体"
            panel.beginSheetModal(for:window) {result in
                guard result == .OK, let selected=panel.url else {replyHandler(NSNull(),nil);return}
                self.nativeRequest("api/desktop/grant",body:["path":selected.path,"kind":action=="directory" ? "directory":"media"],reply:replyHandler)
            }
        } else if action=="reveal", let pid=body["projectId"] as? Int {
            let area=body["area"] as? String ?? "project"
            guard ["project","documents","exports"].contains(area) else {replyHandler(nil,"未知目录");return}
            nativeRequest("api/desktop/projects/\(pid)/directory?area=\(area)") {value,error in
                if let path=(value as? [String:Any])?["path"] as? String {NSWorkspace.shared.selectFile(nil,inFileViewerRootedAtPath:path);replyHandler(true,nil)} else {replyHandler(nil,error)}
            }
        } else {replyHandler(nil,"未知操作")}
    }

    func webView(_ webView:WKWebView, decidePolicyFor navigationAction:WKNavigationAction, decisionHandler:@escaping(WKNavigationActionPolicy)->Void) {
        guard let url=navigationAction.request.url else {decisionHandler(.cancel);return}
        if WorkbenchNavigationPolicy.allows(url:url,baseURL:baseURL,isMainFrame:navigationAction.targetFrame?.isMainFrame != false) {decisionHandler(.allow)}
        else {if navigationAction.navigationType == .linkActivated && ["https","http"].contains(url.scheme ?? "") {NSWorkspace.shared.open(url)};decisionHandler(.cancel)}
    }
    func webView(_ webView:WKWebView, decidePolicyFor navigationResponse:WKNavigationResponse, decisionHandler:@escaping(WKNavigationResponsePolicy)->Void) {
        guard let url=navigationResponse.response.url,
              WorkbenchNavigationPolicy.allows(url:url,baseURL:baseURL,isMainFrame:navigationResponse.isForMainFrame)
        else {decisionHandler(.cancel);return}
        decisionHandler(.allow)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool {true}
    func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {
        terminating=true
        guard let p=process,p.isRunning else {return .terminateNow}
        let pid=p.processIdentifier
        kill(-pid,SIGTERM)
        DispatchQueue.main.asyncAfter(deadline:.now()+2) {
            if p.isRunning {kill(-pid,SIGKILL)}
            NSApp.reply(toApplicationShouldTerminate:true)
        }
        return .terminateLater
    }
}
let app=NSApplication.shared
let delegate=AppDelegate()
app.delegate=delegate
app.setActivationPolicy(.regular)
app.run()
