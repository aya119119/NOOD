// Report — the high-stakes results screen with synced video, timestamped tips, mismatch cards

function Report({ onNav }) {
  const { t, lang } = useT();
  const r = REPORT;
  const [tab, setTab] = useState("overview");
  const [videoT, setVideoT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [doneTips, setDoneTips] = useState([]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setVideoT(t => Math.min(t + 0.25, r.meta.duration_s)), 250);
    return () => clearInterval(id);
  }, [playing, r.meta.duration_s]);

  const jumpTo = (s) => { setVideoT(s); setPlaying(true); };
  const fmt = (s) => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
  const delta = r.overall.score - r.prev.score;
  const sevColor = (s) => s === "high" ? "var(--bad)" : s === "medium" ? "var(--warn)" : "var(--muted)";
  const sevLabel = (s) => s === "high" ? t("rep.severity.high") : s === "medium" ? t("rep.severity.med") : t("rep.severity.low");

  return (
    <div className="fade-in" style={{ padding: "32px 32px 64px", maxWidth: 1280, margin: "0 auto" }}>
      {/* Header strip */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 24, flexWrap: "wrap", gap: 16 }}>
        <div>
          <Eyebrow style={{ marginBottom: 8 }}>{new Date(r.meta.date).toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US", { day: "numeric", month: "long", year: "numeric" })} · {fmt(r.meta.duration_s)}</Eyebrow>
          <h1 className="display" style={{ fontSize: 32, margin: 0 }}>{r.meta.name[lang]}</h1>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Button kind="ghost" icon="ios_share">{t("rep.share")}</Button>
          <Button kind="ghost" icon="download">{t("rep.export")}</Button>
          <Button kind="primary" icon="add" onClick={() => onNav("workspace")}>{t("rep.again")}</Button>
        </div>
      </div>

      {/* HERO: score + 3-pillar + video */}
      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1.4fr", gap: 16, marginBottom: 16 }}>
        {/* Score */}
        <Card padding={28}>
          <div style={{ display: "flex", gap: 28, alignItems: "center", marginBottom: 24 }}>
            <div style={{ position: "relative", width: 140, height: 140, flexShrink: 0 }}>
              <ScoreRing value={r.overall.score} />
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                <div className="num display" style={{ fontSize: 38, fontWeight: 700, lineHeight: 1, letterSpacing: "-0.02em" }}>{r.overall.score.toFixed(1)}</div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>/ 100</div>
              </div>
            </div>
            <div style={{ flex: 1 }}>
              <Eyebrow style={{ marginBottom: 6 }}>{t("rep.score.total")}</Eyebrow>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
                <span className="display" style={{ fontSize: 44, fontWeight: 700 }}>{r.overall.grade}</span>
                <span style={{ color: "var(--muted)", fontSize: 13 }}>{t("rep.grade")}</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "inline-flex", alignItems: "center", gap: 6, color: delta >= 0 ? "var(--good)" : "var(--bad)", fontSize: 12.5, fontWeight: 500 }}>
                  <span className="icon" style={{ fontSize: 16 }}>{delta >= 0 ? "trending_up" : "trending_down"}</span>
                  {delta >= 0 ? "+" : ""}{delta.toFixed(1)} {t("rep.vs.last")}
                </div>
                <div style={{ color: "var(--muted)", fontSize: 12.5 }}>
                  {(r.overall.score - r.avg30 >= 0 ? "+" : "")}{(r.overall.score - r.avg30).toFixed(1)} {t("rep.vs.avg")}
                </div>
              </div>
            </div>
          </div>
          <div className="hr" style={{ marginBottom: 16 }} />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
            {[
              { k: "voice", v: r.components.voice, c: "#3CC58F", label: lang === "fr" ? "Voix" : "Voice" },
              { k: "body",  v: r.components.body,  c: "#5B8DEF", label: lang === "fr" ? "Corps" : "Body" },
              { k: "tone",  v: r.components.tone,  c: "#A06BD8", label: lang === "fr" ? "Ton"   : "Tone" },
              { k: "content", v: r.components.content, c: "#E2A33A", label: lang === "fr" ? "Contenu" : "Content" },
            ].map(p => (
              <div key={p.k}>
                <div className="eyebrow" style={{ marginBottom: 6, color: "var(--muted)" }}>{p.label}</div>
                <div className="num display" style={{ fontSize: 22, fontWeight: 700, marginBottom: 6 }}>{p.v}<span style={{ fontSize: 11, color: "var(--muted)", fontWeight: 400 }}> /100</span></div>
                <div style={{ height: 3, background: "var(--rule-soft)", borderRadius: 999 }}>
                  <div style={{ width: `${p.v}%`, height: "100%", background: p.c, borderRadius: 999 }} />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Video */}
        <Card padding={0} style={{ overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ position: "relative", aspectRatio: "16/9", background: "linear-gradient(135deg, #1a1372 0%, #0F0866 100%)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <button onClick={() => setPlaying(p => !p)} style={{
              width: 64, height: 64, borderRadius: 999, border: 0,
              background: "rgba(255,255,255,0.96)", cursor: "pointer",
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 12px 30px -8px rgba(0,0,0,0.4)",
            }}>
              <span className="icon" style={{ fontSize: 32, color: "var(--ink)" }}>{playing ? "pause" : "play_arrow"}</span>
            </button>
            <div style={{ position: "absolute", left: 16, top: 16, padding: "5px 10px", borderRadius: 999, background: "rgba(0,0,0,0.4)", backdropFilter: "blur(8px)", color: "white", fontSize: 11, fontWeight: 500 }}>
              {fmt(videoT)} / {fmt(r.meta.duration_s)}
            </div>
            {/* moment markers on the video */}
            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 4, background: "rgba(255,255,255,0.18)" }}>
              <div style={{ width: `${(videoT / r.meta.duration_s) * 100}%`, height: "100%", background: "white" }} />
              {r.tone.coaching_tips.map((tip, i) => (
                <div key={i} title={tip.title[lang]} onClick={() => jumpTo(tip.moment_s)} style={{
                  position: "absolute", left: `${(tip.moment_s / r.meta.duration_s) * 100}%`,
                  top: -3, width: 10, height: 10, borderRadius: 999,
                  background: tip.impact === "high" ? "var(--bad)" : tip.impact === "medium" ? "var(--warn)" : "var(--muted-2)",
                  transform: "translateX(-50%)", border: "2px solid white", cursor: "pointer",
                }} />
              ))}
            </div>
          </div>
          {/* timeline lanes (body emotion + voice intensity) */}
          <div style={{ padding: 14, background: "var(--bg-2)" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <Eyebrow>{t("rep.timeline")}</Eyebrow>
              <div className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>3 lanes · {r.body_language.frames_analyzed} frames</div>
            </div>
            <TimelineLanes report={r} videoT={videoT} onSeek={jumpTo} />
          </div>
        </Card>
      </div>

      {/* TABS */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid var(--rule)", marginBottom: 24 }}>
        {[
          { id: "overview", label: t("rep.summary"), icon: "dashboard" },
          { id: "voice",    label: t("rep.section.voice"), icon: "graphic_eq" },
          { id: "body",     label: t("rep.section.body"),  icon: "accessibility_new" },
          { id: "tone",     label: t("rep.section.tone"),  icon: "auto_awesome" },
          { id: "transcript", label: t("rep.transcript"),  icon: "text_snippet" },
        ].map(tg => (
          <button key={tg.id} onClick={() => setTab(tg.id)} style={{
            padding: "14px 18px", border: 0, background: "transparent",
            color: tab === tg.id ? "var(--ink)" : "var(--muted)",
            borderBottom: `2px solid ${tab === tg.id ? "var(--ink)" : "transparent"}`,
            cursor: "pointer", fontFamily: "var(--body)", fontWeight: 500, fontSize: 13.5,
            display: "inline-flex", alignItems: "center", gap: 8, marginBottom: -1,
          }}>
            <span className="icon" style={{ fontSize: 18 }}>{tg.icon}</span>
            {tg.label}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.2fr", gap: 16 }}>
          {/* Coaching tips */}
          <Card padding={24}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <Eyebrow>{t("rep.tips")}</Eyebrow>
              <span className="mono" style={{ color: "var(--muted)" }}>{r.tone.coaching_tips.length}</span>
            </div>
            <div style={{ color: "var(--muted)", fontSize: 13, marginBottom: 16 }}>{t("rep.tips.lede")}</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {r.tone.coaching_tips.map((tip, i) => {
                const done = doneTips.includes(i);
                return (
                  <div key={i} style={{
                    border: "1px solid var(--rule)", borderRadius: 12, padding: 14,
                    opacity: done ? 0.5 : 1, transition: "opacity 0.15s",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 12, marginBottom: 6 }}>
                      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                        <span style={{
                          padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600,
                          textTransform: "uppercase", letterSpacing: "0.08em",
                          color: tip.impact === "high" ? "var(--bad)" : tip.impact === "medium" ? "var(--warn)" : "var(--muted)",
                          background: tip.impact === "high" ? "rgba(192,57,43,0.1)" : tip.impact === "medium" ? "rgba(201,134,21,0.1)" : "var(--rule-soft)",
                        }}>{tip.impact}</span>
                        <button onClick={() => jumpTo(tip.moment_s)} className="mono" style={{
                          border: 0, background: "transparent", color: "var(--ink)", cursor: "pointer",
                          fontSize: 11, padding: 0, display: "inline-flex", alignItems: "center", gap: 4, textDecoration: "underline",
                        }}>
                          <span className="icon" style={{ fontSize: 14 }}>play_circle</span>
                          {fmt(tip.moment_s)}
                        </button>
                      </div>
                      <button onClick={() => setDoneTips(dt => dt.includes(i) ? dt.filter(x => x !== i) : [...dt, i])} title={t("rep.tips.done")}
                        style={{ border: 0, background: "transparent", padding: 4, cursor: "pointer", color: done ? "var(--good)" : "var(--muted-2)" }}>
                        <span className="icon" style={{ fontSize: 18 }}>{done ? "check_circle" : "radio_button_unchecked"}</span>
                      </button>
                    </div>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4, textDecoration: done ? "line-through" : "none" }}>{tip.title[lang]}</div>
                    <div style={{ fontSize: 13, lineHeight: 1.55, color: "var(--ink-soft)" }}>{tip.body[lang]}</div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Mismatches + body summary */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <Card padding={24}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                <Eyebrow>{t("rep.mismatches")} ({r.tone.mismatches.length})</Eyebrow>
                <div style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--good)", fontSize: 12, fontWeight: 500 }}>
                  <span className="icon" style={{ fontSize: 14 }}>check_circle</span>
                  {Math.round(r.tone.fit * 100)}% {t("rep.tone.fit")}
                </div>
              </div>
              {r.tone.mismatches.map((m, i) => (
                <div key={i} onClick={() => jumpTo(m.moment_s)} style={{
                  borderLeft: `2px solid ${sevColor(m.severity)}`, padding: "10px 14px",
                  background: "var(--bg-2)", borderRadius: 4, marginBottom: 8, cursor: "pointer",
                }}>
                  <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8, fontSize: 11 }}>
                    <span style={{ color: sevColor(m.severity), fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>{sevLabel(m.severity)}</span>
                    <span className="mono" style={{ color: "var(--muted)" }}>{fmt(m.moment_s)} · {m.moment_label[lang]}</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 8 }}>
                    <div>
                      <div className="eyebrow" style={{ marginBottom: 4, fontSize: 10 }}>{t("rep.observed")}</div>
                      <div style={{ fontSize: 13, color: "var(--ink-soft)" }}>{m.observed[lang]}</div>
                    </div>
                    <div>
                      <div className="eyebrow" style={{ marginBottom: 4, fontSize: 10 }}>{t("rep.expected")}</div>
                      <div style={{ fontSize: 13, color: "var(--ink)" }}>{m.expected[lang]}</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>{m.reason[lang]}</div>
                </div>
              ))}
            </Card>
          </div>
        </div>
      )}

      {tab === "voice" && (
        <Card padding={24}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
            <Eyebrow>{t("rep.section.voice")}</Eyebrow>
            <span style={{ padding: "4px 10px", borderRadius: 6, background: "var(--bg-2)", fontSize: 12, fontWeight: 600 }}>Grade {r.speech.grade}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
            {Object.entries(r.speech.metrics).map(([k, m]) => (
              <div key={k}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>{m.label[lang]}</div>
                  <div className="num" style={{ fontSize: 14, fontWeight: 600 }}>
                    {typeof m.raw === "number" ? m.raw : m.raw} <span style={{ color: "var(--muted)", fontWeight: 400, fontSize: 11 }}>{m.unit}</span>
                  </div>
                </div>
                {m.range && <MetricBar value={typeof m.raw === "number" ? m.raw : 0} range={m.range} ideal={m.ideal} />}
                <div style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 8, lineHeight: 1.5 }}>{m.feedback[lang]}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {tab === "body" && (
        <Card padding={24}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <Eyebrow>{t("rep.section.body")}</Eyebrow>
            <span className="mono" style={{ color: "var(--muted)" }}>{r.body_language.frames_analyzed} {t("rep.frames")}</span>
          </div>
          <div style={{ marginBottom: 20 }}>
            <div className="eyebrow" style={{ marginBottom: 4 }}>{t("rep.dominant")}</div>
            <div className="display" style={{ fontSize: 28, fontWeight: 700 }}>{r.body_language.dominant} <span style={{ color: "var(--muted)", fontSize: 16, fontWeight: 500 }}>· {r.body_language.dominant_pct}%</span></div>
          </div>
          <StackedBar items={r.body_language.distribution} height={14} />
          <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginTop: 14, marginBottom: 24 }}>
            {r.body_language.distribution.map((it, i) => (
              <div key={i} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: it.color }} />
                <span style={{ color: "var(--ink-soft)" }}>{it.label}</span>
                <span className="num" style={{ color: "var(--muted)" }}>{it.pct.toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <div style={{ padding: 16, background: "var(--bg-2)", borderRadius: 10, fontSize: 13.5, lineHeight: 1.55, color: "var(--ink-soft)" }}>
            <span className="icon" style={{ fontSize: 16, color: "var(--ink)", marginRight: 8, verticalAlign: "middle" }}>auto_awesome</span>
            {r.body_language.interpretation[lang]}
          </div>
        </Card>
      )}

      {tab === "tone" && (
        <Card padding={24}>
          <Eyebrow style={{ marginBottom: 16 }}>{t("rep.section.tone")}</Eyebrow>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
            <div style={{ padding: 16, background: "var(--bg-2)", borderRadius: 10 }}>
              <div className="eyebrow" style={{ marginBottom: 6 }}>{t("rep.tone.topic")}</div>
              <div style={{ fontSize: 14, color: "var(--ink)" }}>{r.tone.topic[lang]}</div>
            </div>
            <div style={{ padding: 16, background: "var(--bg-2)", borderRadius: 10 }}>
              <div className="eyebrow" style={{ marginBottom: 6 }}>{t("rep.tone.context")}</div>
              <div style={{ fontSize: 14, color: "var(--ink)" }}>{r.tone.context[lang]}</div>
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: 18, border: "1px solid var(--rule)", borderRadius: 10 }}>
            <div>
              <div className="eyebrow" style={{ marginBottom: 4 }}>{t("rep.tone.fit")}</div>
              <div style={{ color: "var(--good)", fontWeight: 600, fontSize: 16 }}>{r.tone.fit_label[lang]}</div>
            </div>
            <div className="num display" style={{ fontSize: 36, fontWeight: 700 }}>{Math.round(r.tone.fit * 100)}%</div>
          </div>
        </Card>
      )}

      {tab === "transcript" && (
        <Card padding={24}>
          <Eyebrow style={{ marginBottom: 14 }}>{t("rep.transcript")}</Eyebrow>
          <div style={{ fontSize: 15, lineHeight: 1.7, color: "var(--ink-soft)" }}>
            {r.speech.transcript_preview[lang]}
          </div>
        </Card>
      )}
    </div>
  );
}

function TimelineLanes({ report, videoT, onSeek }) {
  const dur = report.meta.duration_s;
  const lanes = [
    { label: "BODY", values: report.body_language.timeline.map(f => ({ t: f.t, c: report.body_language.distribution.find(d => d.label === f.emotion)?.color || "#999" })) },
    { label: "ENERGY", values: Array.from({ length: 60 }, (_, i) => ({ t: i * dur / 60, c: `rgba(15,8,102,${0.15 + Math.abs(Math.sin(i*0.5)) * 0.7})` })) },
    { label: "PITCH", values: Array.from({ length: 60 }, (_, i) => ({ t: i * dur / 60, c: `rgba(107,92,255,${0.12 + Math.abs(Math.cos(i*0.4)) * 0.65})` })) },
  ];
  return (
    <div onClick={(e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      onSeek(x * dur);
    }} style={{ cursor: "pointer", position: "relative" }}>
      {lanes.map((lane, li) => (
        <div key={li} style={{ display: "grid", gridTemplateColumns: "44px 1fr", gap: 8, alignItems: "center", marginBottom: 4 }}>
          <span className="mono" style={{ fontSize: 9, color: "var(--muted)" }}>{lane.label}</span>
          <div style={{ display: "flex", gap: 1, height: 14 }}>
            {lane.values.map((v, i) => <div key={i} style={{ flex: 1, background: v.c, borderRadius: 1 }} />)}
          </div>
        </div>
      ))}
      <div style={{
        position: "absolute", left: `calc(44px + 8px + ${(videoT / dur) * 100}% * (100% - 52px) / 100% - 1px)`,
        top: 0, bottom: 0, width: 2, background: "var(--ink)", pointerEvents: "none",
      }} />
    </div>
  );
}

window.Report = Report;
