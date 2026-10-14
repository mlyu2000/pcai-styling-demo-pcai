"""UI end-to-end for the PCAI styling demo: drives headless Chrome over CDP,
uploads a real front-view photo, walks W1 stage4 -> gate -> stage5 -> video,
saves a full-page screenshot per step, and dumps app state as JSON.

Usage: python scripts/ui_e2e.py <image.(jpg|png)> <outdir>
"""
import base64, http.client, json, os, shutil, subprocess, sys, time

import websocket

CHROME = r"C:\Users\ml\.agent-browser\browsers\chrome-154.0.8037.57\chrome.exe"
URL = "https://styling-demo.aie.cs1.ctc.sg.lab/"
PORT = 9333


class Tab:
    def __init__(self, ws):
        self.ws = ws
        self.mid = 0

    def call(self, method, params=None):
        self.mid += 1
        self.ws.send(json.dumps({"id": self.mid, "method": method,
                                 "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate",
                      {"expression": expr, "returnByValue": True,
                       "awaitPromise": True})
        if r.get("exceptionDetails"):
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:300])
        return r.get("result", {}).get("value")

    def shot(self, path, scroll_sel=None):
        if scroll_sel:
            self.js(f'(()=>{{const e=document.querySelector("{scroll_sel}");'
                    f'if(e)e.scrollIntoView({{block:"start"}})}})()')
        r = self.call("Page.captureScreenshot",
                      {"format": "png", "captureBeyondViewport": True})
        open(path, "wb").write(base64.b64decode(r["data"]))
        print("shot:", os.path.basename(path), os.path.getsize(path), flush=True)

    def set_files(self, selector, paths):
        doc = self.call("DOM.getDocument")["root"]["nodeId"]
        nid = self.call("DOM.querySelector",
                        {"nodeId": doc, "selector": selector})["nodeId"]
        if not nid:
            raise RuntimeError(f"no node for {selector}")
        self.call("DOM.setFileInputFiles", {"nodeId": nid, "files": paths})


def wait_until(tab, expr, timeout_s, poll_s=5):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            last = tab.js(expr)
        except Exception as e:
            last = f"ERR {type(e).__name__}"
        if last:
            return last
        print(f"  waiting {int(time.time()-t0)}s …", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"timed out: {expr[:90]} (last={last})")


def main():
    img, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    profile = os.path.join(outdir, "profile")
    shutil.rmtree(profile, ignore_errors=True)
    proc = subprocess.Popen(
        [CHROME, f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
         "--headless=new", "--no-sandbox", "--disable-gpu",
         "--window-size=1500,3400", "--ignore-certificate-errors",
         "--user-data-dir=" + profile, "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tabs = None
        for _ in range(30):
            try:
                c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
                c.request("GET", "/json")
                tabs = json.loads(c.getresponse().read())
                if tabs:
                    break
            except Exception:
                time.sleep(1)
        page = next(t for t in tabs if t["type"] == "page")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"],
                                         timeout=120, max_size=None)
        tab = Tab(ws)
        tab.call("Page.enable"); tab.call("Runtime.enable"); tab.call("DOM.enable")
        tab.call("Page.navigate", {"url": URL})
        time.sleep(6)
        # alerts block the JS thread under CDP — neutralize them first
        tab.js("window.alert=(m)=>{window.__lastAlert=m}; window.confirm=()=>true")

        tab.js('showTab("workflow")')
        time.sleep(1)
        tab.shot(os.path.join(outdir, "00_workflow_tab.png"), "#wfsec1")

        tab.set_files("#fS1", [os.path.abspath(img)])
        wait_until(tab, 'typeof s1B64==="string" && s1B64.length>1000', 30)
        tab.shot(os.path.join(outdir, "01_uploaded.png"), "#wfsec1")

        # W1: stage 4 — AI base sketch (front + rear + colorways)
        tab.js('document.querySelector("#s4run").click()')
        wait_until(tab, '/Done in/.test(document.querySelector("#s4status").textContent)', 420)
        print("stage4 sets rendered:",
              tab.js('document.querySelectorAll("#s4out .setgroup").length'), flush=True)
        tab.shot(os.path.join(outdir, "02_stage4_sketches.png"), "#wfsec1")

        # pick the first image of Set 1 (the pick listener is on the img)
        tab.js('document.querySelector("#s4out .setgroup .cell img").click()')
        wait_until(tab, "!!wf.sketchFront", 15)

        # W2: designer gate — simulated Photoshop pass
        tab.js('document.querySelector("#mockDesign").click()')
        wait_until(tab, "!!wf.designImage && wf.useMockDesign", 15)
        tab.shot(os.path.join(outdir, "03_gate_design.png"), "#wfsec2")

        # W3: sample scene + stage 5 compose/quick-viz
        tab.js('(()=>{const b=[...document.querySelectorAll("#wfsec3 button")]'
               '.find(x=>/sample scene/i.test(x.textContent)); if(b)b.click(); return !!b})()')
        time.sleep(1)
        tab.js('document.querySelector("#s5run").click()')
        wait_until(tab, '/Done in/.test(document.querySelector("#s5status").textContent)', 420)
        tab.shot(os.path.join(outdir, "04_stage5_quickviz.png"), "#wfsec3")

        # pick the MAIN still (cells use class "cell"; img carries the listener)
        cap = tab.js('(()=>{const cs=[...document.querySelectorAll("#s5out .cell")];'
                     'const c=cs.find(x=>/main/i.test(x.textContent))||cs[1]||cs[0];'
                     'c.querySelector("img").click(); return c.textContent.slice(0,30)})()')
        print("picked viz cell:", cap, flush=True)
        wait_until(tab, "!!wf.finalPick", 15)

        # W4: stage 3 video from the chosen quick-viz still
        tab.js('document.querySelector("#s3run").click()')
        wait_until(tab, "!!(wf.s3 && wf.s3.videoUrl)", 480)
        tab.shot(os.path.join(outdir, "05_stage3_video.png"), "#wfsec4")

        state = tab.js('JSON.stringify({sets:document.querySelectorAll("#s4out .setgroup").length,'
                       ' colorways:(wf.s4.colorways||[]).map(c=>c.color),'
                       ' designSource:wf.designSource,'
                       ' viz:!!(wf.s5&&wf.s5.main), video:wf.s3&&wf.s3.videoUrl})')
        print("FINAL_STATE:", state, flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
