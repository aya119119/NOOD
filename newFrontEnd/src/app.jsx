// App shell — routing + tweaks

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "showDevPill": true,
  "defaultLang": "fr",
  "startScreen": "landing"
}/*EDITMODE-END*/;

function App() {
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [route, setRoute] = useState(tweaks.startScreen || "landing");
  const [signedIn, setSignedIn] = useState(false);
  const [fileState, setFileState] = useState(null);

  const onNav = useCallback((to, opts) => {
    if (opts?.signedIn) setSignedIn(true);
    if (to === "auth-signin") { setRoute("auth"); window.__authMode = "signin"; }
    else if (to === "auth-signup") { setRoute("auth"); window.__authMode = "signup"; }
    else { setRoute(to); }
    if (to === "landing") setSignedIn(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  }, []);

  const isAppShell = ["workspace","processing","report","history","account"].includes(route);
  const showHeader = route !== "auth";

  return (
    <I18nProvider defaultLang={tweaks.defaultLang || "fr"}>
      {tweaks.showDevPill && <DevToolPill />}
      {showHeader && <Header variant={isAppShell ? "app" : "marketing"} route={route} signedIn={signedIn} onNav={onNav} />}
      {route === "landing"    && <Landing onNav={onNav} />}
      {route === "auth"       && <Auth mode={window.__authMode || "signin"} onNav={onNav} />}
      {route === "workspace"  && <Workspace onNav={onNav} fileState={fileState} setFileState={setFileState} />}
      {route === "processing" && <Processing onNav={onNav} />}
      {route === "report"     && <Report onNav={onNav} />}
      {route === "history"    && <History onNav={onNav} />}
      {route === "account"    && <AccountStub onNav={onNav} />}

      <TweaksPanel title="Tweaks">
        <TweakSection label="Demo controls">
          <TweakSelect label="Jump to screen" value={route} onChange={(v) => onNav(v)} options={[
            { value: "landing", label: "Landing" },
            { value: "auth", label: "Auth" },
            { value: "workspace", label: "Workspace" },
            { value: "processing", label: "Processing" },
            { value: "report", label: "Report" },
            { value: "history", label: "History" },
          ]} />
          <TweakToggle label="Signed in (header state)" value={signedIn} onChange={setSignedIn} />
        </TweakSection>
        <TweakSection label="Display">
          <TweakToggle label="Show dev tool pill" value={tweaks.showDevPill} onChange={(v) => setTweak("showDevPill", v)} />
          <TweakRadio label="Default language" value={tweaks.defaultLang} onChange={(v) => setTweak("defaultLang", v)} options={[
            { value: "fr", label: "FR" },
            { value: "en", label: "EN" },
          ]} />
        </TweakSection>
      </TweaksPanel>
    </I18nProvider>
  );
}

function AccountStub({ onNav }) {
  return (
    <div style={{ padding: "64px 32px", maxWidth: 720, margin: "0 auto" }}>
      <h1 className="display" style={{ fontSize: 36 }}>Account</h1>
      <p style={{ color: "var(--muted)" }}>Settings & billing live here.</p>
      <Button kind="ghost" onClick={() => onNav("history")}>Back to history</Button>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
