// History — past sessions list with trend chart

function History({ onNav }) {
  const { t, lang } = useT();
  const trend = HISTORY.map(h => h.score).reverse();
  const avg = (HISTORY.reduce((s, h) => s + h.score, 0) / HISTORY.length).toFixed(1);

  return (
    <div className="fade-in" style={{ padding: "48px 32px 64px", maxWidth: 1280, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 32, flexWrap: "wrap", gap: 16 }}>
        <div>
          <Eyebrow style={{ marginBottom: 8 }}>{HISTORY.length} {lang === "fr" ? "sessions" : "sessions"}</Eyebrow>
          <h1 className="display" style={{ fontSize: 36, margin: "0 0 6px" }}>{t("hist.title")}</h1>
          <p style={{ color: "var(--muted)", margin: 0, fontSize: 14 }}>{t("hist.lede")}</p>
        </div>
        <Button kind="primary" icon="add" onClick={() => onNav("workspace")}>{t("hist.new")}</Button>
      </div>

      {/* Trend strip */}
      <Card padding={24} style={{ marginBottom: 16 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 2fr", gap: 24, alignItems: "center" }}>
          <div>
            <Eyebrow style={{ marginBottom: 6 }}>{t("hist.trend")}</Eyebrow>
            <div className="display num" style={{ fontSize: 36, fontWeight: 700 }}>{avg}</div>
          </div>
          <div>
            <Eyebrow style={{ marginBottom: 6 }}>{lang === "fr" ? "Meilleur score" : "Best score"}</Eyebrow>
            <div className="display num" style={{ fontSize: 24, fontWeight: 700 }}>{Math.max(...HISTORY.map(h => h.score))}</div>
          </div>
          <div>
            <Eyebrow style={{ marginBottom: 6 }}>{lang === "fr" ? "Évolution" : "Change"}</Eyebrow>
            <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--good)" }}>
              <span className="icon">trending_up</span>
              <span className="num display" style={{ fontSize: 24, fontWeight: 700 }}>+14.5</span>
            </div>
          </div>
          <div style={{ height: 60 }}>
            <Sparkline values={trend} width={400} height={60} />
          </div>
        </div>
      </Card>

      {/* Sessions list */}
      <Card padding={0}>
        <div style={{ padding: "12px 24px", display: "grid", gridTemplateColumns: "60px 1fr 100px 120px 100px 100px 60px", gap: 16, borderBottom: "1px solid var(--rule)", alignItems: "center" }}>
          {[
            lang === "fr" ? "Date" : "Date",
            lang === "fr" ? "Session" : "Session",
            lang === "fr" ? "Type" : "Kind",
            lang === "fr" ? "Durée" : "Duration",
            lang === "fr" ? "Score" : "Score",
            lang === "fr" ? "Note" : "Grade",
            "",
          ].map((h, i) => <div key={i} className="eyebrow">{h}</div>)}
        </div>
        {HISTORY.map((s, i) => (
          <div key={s.id} onClick={() => onNav("report")} style={{
            padding: "16px 24px", display: "grid", gridTemplateColumns: "60px 1fr 100px 120px 100px 100px 60px", gap: 16,
            borderBottom: i < HISTORY.length - 1 ? "1px solid var(--rule-soft)" : "none",
            alignItems: "center", cursor: "pointer", transition: "background 0.12s",
          }}
            onMouseEnter={e => e.currentTarget.style.background = "var(--hover)"}
            onMouseLeave={e => e.currentTarget.style.background = "transparent"}
          >
            <div className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>{s.date.slice(5)}</div>
            <div style={{ fontWeight: 500, fontSize: 14 }}>{s.name[lang]}</div>
            <div><span style={{ padding: "2px 8px", border: "1px solid var(--rule)", borderRadius: 999, fontSize: 11, color: "var(--muted)" }}>{s.kind[lang]}</span></div>
            <div className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>{Math.floor(s.duration/60)}:{String(s.duration%60).padStart(2,'0')}</div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 50, height: 4, background: "var(--rule-soft)", borderRadius: 999 }}>
                <div style={{ width: `${s.score}%`, height: "100%", background: s.score >= 70 ? "var(--good)" : s.score >= 55 ? "var(--warn)" : "var(--bad)", borderRadius: 999 }} />
              </div>
              <span className="num" style={{ fontWeight: 600, fontSize: 13 }}>{s.score}</span>
            </div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{s.grade}</div>
            <span className="icon" style={{ fontSize: 18, color: "var(--muted-2)" }}>chevron_right</span>
          </div>
        ))}
      </Card>
    </div>
  );
}

window.History = History;
