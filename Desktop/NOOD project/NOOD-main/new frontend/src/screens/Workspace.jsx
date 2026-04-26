// Workspace — upload + context capture (replaces "AI Assistant" with structured context)

function Workspace({ onNav, fileState, setFileState }) {
  const { t, lang } = useT();
  const [file, setFile] = useState(fileState?.file || null);
  const [recording, setRecording] = useState(false);
  const [kind, setKind] = useState(fileState?.kind || 0);
  const [audience, setAudience] = useState(fileState?.audience || 0);
  const [goal, setGoal] = useState(fileState?.goal || 0);
  const [script, setScript] = useState(fileState?.script || "");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const fakeFile = { name: "demo_day_pitch_v3.mp4", size: "47.2 MB", duration: "3:07" };

  const onPick = () => inputRef.current?.click();
  const onFileSelected = (f) => { if (f) setFile({ name: f.name, size: `${(f.size/1024/1024).toFixed(1)} MB`, duration: "—" }); };
  const useDemo = () => setFile(fakeFile);

  const start = () => {
    setFileState({ file, kind, audience, goal, script });
    onNav("processing");
  };

  return (
    <div className="fade-in" style={{ padding: "48px 32px 64px", maxWidth: 1280, margin: "0 auto" }}>
      <div style={{ marginBottom: 32 }}>
        <Eyebrow style={{ marginBottom: 10 }}>{lang === "fr" ? "Nouvelle analyse" : "New analysis"}</Eyebrow>
        <h1 className="display" style={{ fontSize: 42, margin: 0 }}>
          {t("ws.title.a")} <span className="display-italic" style={{ color: "var(--ink-soft)" }}>{t("ws.title.b")}</span>
        </h1>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 16 }}>
        {/* UPLOAD */}
        <Card padding={0} style={{ overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--rule)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <Eyebrow style={{ marginBottom: 4 }}>01 · {lang === "fr" ? "Source" : "Source"}</Eyebrow>
              <div className="h3">{lang === "fr" ? "Vidéo à analyser" : "Video to analyze"}</div>
            </div>
            {!file && (
              <button onClick={useDemo} className="mono" style={{
                border: "1px solid var(--rule)", background: "white", padding: "5px 10px",
                borderRadius: 999, cursor: "pointer", color: "var(--muted)", fontSize: 11,
              }}>{lang === "fr" ? "essayer un exemple" : "try a sample"}</button>
            )}
          </div>

          {!file ? (
            <div style={{ padding: 24, flex: 1 }}>
              <div
                onDragOver={e => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={e => { e.preventDefault(); setDrag(false); onFileSelected(e.dataTransfer.files?.[0]); }}
                style={{
                  border: `1.5px dashed ${drag ? "var(--ink)" : "var(--rule)"}`,
                  background: drag ? "var(--hover)" : "var(--bg-2)",
                  borderRadius: 14, padding: "48px 24px", textAlign: "center",
                  transition: "all 0.15s",
                }}>
                <div style={{
                  width: 56, height: 56, borderRadius: 999, background: "white",
                  border: "1px solid var(--rule)", display: "inline-flex",
                  alignItems: "center", justifyContent: "center", marginBottom: 20,
                }}>
                  <span className="icon" style={{ fontSize: 28, color: "var(--ink)" }}>cloud_upload</span>
                </div>
                <div className="h3" style={{ marginBottom: 4 }}>{t("ws.upload.drop")}</div>
                <div style={{ color: "var(--muted)", fontSize: 13, marginBottom: 22 }}>{t("ws.upload.or")}</div>
                <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 24 }}>
                  <Button kind="primary" icon="folder_open" onClick={onPick}>{t("ws.upload.browse")}</Button>
                  <Button kind="ghost" icon={recording ? "stop_circle" : "videocam"} onClick={() => setRecording(r => !r)}>
                    {recording ? (lang === "fr" ? "Arrêter" : "Stop") : t("ws.upload.record")}
                  </Button>
                </div>
                <div className="mono" style={{ color: "var(--muted-2)", fontSize: 11 }}>{t("ws.upload.formats")}</div>
                <input ref={inputRef} type="file" accept="video/*" hidden onChange={e => onFileSelected(e.target.files?.[0])} />
              </div>
            </div>
          ) : (
            <div style={{ padding: 24, flex: 1 }}>
              <div style={{ display: "flex", gap: 16, padding: 16, border: "1px solid var(--rule)", borderRadius: 12, marginBottom: 16, alignItems: "center" }}>
                <div style={{
                  width: 64, height: 64, borderRadius: 10, background: "var(--ink)", color: "white",
                  display: "inline-flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                }}>
                  <span className="icon" style={{ fontSize: 28 }}>movie</span>
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</div>
                  <div className="mono" style={{ color: "var(--muted)", marginTop: 4 }}>
                    {file.size} · {file.duration} · MP4
                  </div>
                </div>
                <button onClick={() => setFile(null)} style={{
                  border: 0, background: "transparent", padding: 8, cursor: "pointer", color: "var(--muted)",
                }}><span className="icon">close</span></button>
              </div>

              {/* fake waveform preview */}
              <div style={{ padding: 16, background: "var(--bg-2)", borderRadius: 12, marginBottom: 20 }}>
                <Eyebrow style={{ marginBottom: 10 }}>{lang === "fr" ? "Aperçu audio" : "Audio preview"}</Eyebrow>
                <div style={{ display: "flex", alignItems: "center", gap: 1, height: 40 }}>
                  {Array.from({ length: 80 }).map((_, i) => {
                    const h = 10 + Math.abs(Math.sin(i * 0.4) * 22) + Math.abs(Math.cos(i * 0.17) * 8);
                    return <div key={i} style={{ flex: 1, background: i < 8 ? "var(--ink)" : "var(--muted-2)", borderRadius: 1, height: `${h}px`, opacity: i < 8 ? 1 : 0.5 }} />;
                  })}
                </div>
                <div className="mono" style={{ color: "var(--muted)", fontSize: 10, marginTop: 6, display: "flex", justifyContent: "space-between" }}>
                  <span>0:00</span><span>{file.duration}</span>
                </div>
              </div>

              <Button kind="primary" size="lg" icon="play_arrow" onClick={start} style={{ width: "100%" }}>
                {t("ws.upload.analyze")}
              </Button>
            </div>
          )}

          <div style={{ padding: "14px 24px", background: "var(--bg-2)", borderTop: "1px solid var(--rule)", display: "flex", gap: 18, fontSize: 11.5, color: "var(--muted)", flexWrap: "wrap" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span className="icon" style={{ fontSize: 14, color: "var(--good)" }}>lock</span>{t("ws.priv.encrypt")}</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span className="icon" style={{ fontSize: 14 }}>delete_history</span>{t("ws.priv.delete")}</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span className="icon" style={{ fontSize: 14 }}>dns</span>{t("ws.priv.local")}</span>
          </div>
        </Card>

        {/* CONTEXT */}
        <Card padding={0} style={{ overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--rule)" }}>
            <Eyebrow style={{ marginBottom: 4 }}>02 · {lang === "fr" ? "Contexte" : "Context"}</Eyebrow>
            <div className="h3">{t("ws.context.title")}</div>
            <div style={{ color: "var(--muted)", fontSize: 13, marginTop: 4 }}>{t("ws.context.lede")}</div>
          </div>
          <div style={{ padding: 24, flex: 1, display: "flex", flexDirection: "column", gap: 18 }}>
            <ChipGroup label={t("ws.context.kind")} options={STRINGS["ws.context.kind.options"][lang]} value={kind} onChange={setKind} />
            <ChipGroup label={t("ws.context.audience")} options={STRINGS["ws.context.audience.options"][lang]} value={audience} onChange={setAudience} />
            <ChipGroup label={t("ws.context.goal")} options={STRINGS["ws.context.goal.options"][lang]} value={goal} onChange={setGoal} />

            <div>
              <Eyebrow style={{ marginBottom: 6 }}>{t("ws.context.script")}</Eyebrow>
              <textarea value={script} onChange={e => setScript(e.target.value)} placeholder={t("ws.context.script.ph")}
                style={{
                  width: "100%", minHeight: 96, padding: 14, border: "1px solid var(--rule)", borderRadius: 10,
                  background: "white", fontSize: 13, fontFamily: "var(--body)", color: "var(--ink)",
                  outline: "none", resize: "vertical", lineHeight: 1.55,
                }}
                onFocus={e => e.currentTarget.style.borderColor = "var(--ink)"}
                onBlur={e => e.currentTarget.style.borderColor = "var(--rule)"}
              />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function ChipGroup({ label, options, value, onChange }) {
  return (
    <div>
      <Eyebrow style={{ marginBottom: 8 }}>{label}</Eyebrow>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {options.map((o, i) => (
          <button key={i} onClick={() => onChange(i)} style={{
            padding: "7px 13px", borderRadius: 999, fontSize: 12.5, fontWeight: 500,
            border: `1px solid ${value === i ? "var(--ink)" : "var(--rule)"}`,
            background: value === i ? "var(--ink)" : "white",
            color: value === i ? "white" : "var(--ink-soft)",
            cursor: "pointer", fontFamily: "var(--body)", transition: "all 0.12s",
          }}>{o}</button>
        ))}
      </div>
    </div>
  );
}

window.Workspace = Workspace;
