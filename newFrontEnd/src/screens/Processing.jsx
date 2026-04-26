// Processing — rich live feedback (parallel tracks, streaming transcript)

function Processing({ onNav }) {
  const { t, lang } = useT();
  const [elapsed, setElapsed] = useState(0);
  const total = 60; // seconds simulated

  const stages = [
    { key: "audio",   icon: "graphic_eq",       at: 0,    dur: 4,  track: "voice", label: t("proc.stage.audio") },
    { key: "vad",     icon: "pause",            at: 4,    dur: 6,  track: "voice", label: t("proc.stage.vad") },
    { key: "asr",     icon: "transcribe",       at: 10,   dur: 14, track: "voice", label: t("proc.stage.asr") },
    { key: "prosody", icon: "tune",             at: 24,   dur: 8,  track: "voice", label: t("proc.stage.prosody") },
    { key: "vocal",   icon: "sentiment_satisfied", at: 32, dur: 6,  track: "voice", label: t("proc.stage.vocal") },
    { key: "tone",    icon: "auto_awesome",     at: 38,   dur: 12, track: "voice", label: t("proc.stage.tone") },
    { key: "body",    icon: "accessibility_new",at: 0,    dur: 36, track: "body",  label: t("proc.stage.body") },
    { key: "score",   icon: "scoreboard",       at: 50,   dur: 10, track: "all",   label: t("proc.stage.score") },
  ];

  useEffect(() => {
    const id = setInterval(() => setElapsed(e => {
      if (e >= total) { clearInterval(id); setTimeout(() => onNav("report"), 500); return total; }
      return e + 0.25;
    }), 100);
    return () => clearInterval(id);
  }, []);

  const progress = (elapsed / total) * 100;
  const fmt = (s) => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;

  // streaming transcript
  const fullTranscript = lang === "fr"
    ? "Bonjour à tous. Aujourd'hui je vais vous présenter NOOD, une plateforme qui transforme la manière dont nous nous entraînons à parler en public. Le problème est simple. La plupart d'entre nous improvisent leurs présentations sans aucun retour structuré."
    : "Hello everyone. Today I'm going to present NOOD, a platform that transforms how we train ourselves to speak in public. The problem is simple. Most of us improvise our presentations without any structured feedback.";
  const visibleChars = Math.min(fullTranscript.length, Math.floor((elapsed - 10) * 8));
  const transcript = visibleChars > 0 ? fullTranscript.slice(0, visibleChars) : "";

  return (
    <div className="fade-in" style={{ padding: "48px 32px 64px", maxWidth: 1280, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 32, flexWrap: "wrap", gap: 16 }}>
        <div>
          <Eyebrow style={{ marginBottom: 10 }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--good)", animation: "pulse 1.4s infinite" }} />
              {lang === "fr" ? "En cours" : "In progress"}
            </span>
          </Eyebrow>
          <h1 className="display" style={{ fontSize: 36, margin: "0 0 6px" }}>{t("proc.title")}</h1>
          <p style={{ color: "var(--muted)", fontSize: 14, margin: 0 }}>{t("proc.lede")}</p>
        </div>
        <div style={{ display: "flex", gap: 32, alignItems: "center" }}>
          <div>
            <Eyebrow style={{ marginBottom: 4 }}>{t("proc.elapsed")}</Eyebrow>
            <div className="num display" style={{ fontSize: 24, fontWeight: 600 }}>{fmt(elapsed)}</div>
          </div>
          <div>
            <Eyebrow style={{ marginBottom: 4 }}>{t("proc.eta")}</Eyebrow>
            <div className="num display" style={{ fontSize: 24, fontWeight: 600, color: "var(--muted)" }}>~{fmt(total - elapsed)}</div>
          </div>
          <Button kind="ghost" icon="close" onClick={() => onNav("workspace")}>{t("proc.cancel")}</Button>
        </div>
      </div>

      {/* Master progress */}
      <Card padding={20} style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <Eyebrow>{lang === "fr" ? "Progression globale" : "Overall progress"}</Eyebrow>
          <div className="num" style={{ fontSize: 13, fontWeight: 600 }}>{Math.floor(progress)}%</div>
        </div>
        <div style={{ height: 6, background: "var(--rule-soft)", borderRadius: 999, overflow: "hidden" }}>
          <div style={{ width: `${progress}%`, height: "100%", background: "linear-gradient(90deg, var(--ink) 0%, var(--accent) 100%)", borderRadius: 999, transition: "width 0.3s" }} />
        </div>
      </Card>

      {/* PARALLEL TRACKS */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* Voice + Tone track */}
        <Card padding={20}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span className="icon" style={{ fontSize: 22, color: "var(--ink)" }}>graphic_eq</span>
              <div className="h3">{t("proc.parallel.voice")}</div>
            </div>
            <MiniWaveform bars={8} active={elapsed < total} />
          </div>
          {stages.filter(s => s.track === "voice").map(s => <StageRow key={s.key} stage={s} elapsed={elapsed} />)}
        </Card>

        {/* Body track */}
        <Card padding={20}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span className="icon" style={{ fontSize: 22, color: "var(--ink)" }}>accessibility_new</span>
              <div className="h3">{t("proc.parallel.body")}</div>
            </div>
            <span className="mono" style={{ color: "var(--muted)" }}>~5 fps</span>
          </div>
          {stages.filter(s => s.track === "body").map(s => <StageRow key={s.key} stage={s} elapsed={elapsed} />)}

          {/* fake emotion frame ticker */}
          <div className="hr" style={{ margin: "16px 0" }} />
          <Eyebrow style={{ marginBottom: 8 }}>{lang === "fr" ? "Émotions détectées" : "Emotions detected"}</Eyebrow>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {Array.from({ length: Math.floor(Math.min(elapsed, 36) * 1.6) }).map((_, i) => {
              const colors = ["#3CC58F","#5B8DEF","#E2A33A","#A06BD8","#3CC58F","#3CC58F"];
              return <div key={i} style={{ width: 6, height: 14, borderRadius: 1, background: colors[i % colors.length], opacity: 0.85 }} />;
            })}
          </div>
        </Card>
      </div>

      {/* Live transcript */}
      <Card padding={20} style={{ marginBottom: 16, minHeight: 140 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <Eyebrow>{t("proc.transcript.live")}</Eyebrow>
          {elapsed > 10 && <div className="num mono" style={{ color: "var(--muted)", fontSize: 11 }}>{Math.floor(transcript.split(" ").length / Math.max(1, elapsed/60) * 1)} mots</div>}
        </div>
        <div style={{ fontSize: 14.5, lineHeight: 1.65, color: "var(--ink-soft)", fontFamily: "var(--body)", minHeight: 60 }}>
          {transcript || <span style={{ color: "var(--muted-2)" }}>{lang === "fr" ? "En attente de la transcription…" : "Waiting for transcription…"}</span>}
          {elapsed > 10 && elapsed < total && <span style={{ display: "inline-block", width: 7, height: 16, background: "var(--ink)", marginLeft: 2, verticalAlign: "middle", animation: "pulse 0.9s infinite" }} />}
        </div>
      </Card>

      {/* Tip */}
      <Card padding={20} style={{ background: "var(--ink)", color: "white", border: "1px solid var(--ink)" }}>
        <div style={{ display: "flex", gap: 16, alignItems: "start" }}>
          <span className="icon" style={{ fontSize: 24, color: "white", opacity: 0.6 }}>lightbulb</span>
          <div>
            <div className="eyebrow" style={{ color: "rgba(255,255,255,0.6)", marginBottom: 6 }}>{t("proc.tip.title")}</div>
            <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{t("proc.tip.body")}</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function StageRow({ stage, elapsed }) {
  const status = elapsed < stage.at ? "pending" : elapsed > stage.at + stage.dur ? "done" : "active";
  const localProgress = status === "done" ? 100 : status === "pending" ? 0 :
    Math.min(100, ((elapsed - stage.at) / stage.dur) * 100);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "24px 1fr 60px", gap: 12, alignItems: "center", padding: "10px 0", borderBottom: "1px solid var(--rule-soft)" }}>
      <span className="icon" style={{
        fontSize: 18,
        color: status === "done" ? "var(--good)" : status === "active" ? "var(--ink)" : "var(--muted-2)",
      }}>{status === "done" ? "check_circle" : stage.icon}</span>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: status === "pending" ? "var(--muted-2)" : "var(--ink)", marginBottom: 4 }}>
          {stage.label}
        </div>
        <div style={{ height: 3, background: "var(--rule-soft)", borderRadius: 999, overflow: "hidden" }}>
          <div style={{ width: `${localProgress}%`, height: "100%", background: status === "done" ? "var(--good)" : "var(--ink)", borderRadius: 999, transition: "width 0.3s" }} />
        </div>
      </div>
      <div className="mono" style={{ fontSize: 11, color: "var(--muted)", textAlign: "right" }}>
        {status === "active" ? `${Math.floor(localProgress)}%` : status === "done" ? "✓" : "—"}
      </div>
    </div>
  );
}

window.Processing = Processing;
