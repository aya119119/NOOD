// Landing page — premium minimal, bilingual

function Landing({ onNav }) {
  const { t, lang } = useT();
  const voiceItems = STRINGS["landing.dim.voice.items"][lang];
  const bodyItems = STRINGS["landing.dim.body.items"][lang];
  const contentItems = STRINGS["landing.dim.content.items"][lang];

  return (
    <div className="fade-in">
      {/* HERO */}
      <section style={{ position: "relative", padding: "88px 32px 96px", overflow: "hidden" }}>
        {/* Lavender bloom backdrop */}
        <div aria-hidden style={{
          position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0,
          background: `
            radial-gradient(ellipse 70% 60% at 75% 35%, rgba(160, 107, 216, 0.28) 0%, rgba(160, 107, 216, 0) 60%),
            radial-gradient(ellipse 60% 50% at 90% 60%, rgba(107, 92, 255, 0.22) 0%, rgba(107, 92, 255, 0) 65%),
            radial-gradient(ellipse 80% 70% at 30% 80%, rgba(91, 141, 239, 0.14) 0%, rgba(91, 141, 239, 0) 60%)
          `
        }} />
        {/* Subtle arcs */}
        <svg aria-hidden viewBox="0 0 1280 800" preserveAspectRatio="xMidYMid slice" style={{
          position: "absolute", inset: 0, width: "100%", height: "100%", zIndex: 0, opacity: 0.5, pointerEvents: "none"
        }}>
          <defs>
            <linearGradient id="arcGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#A06BD8" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#6b5cff" stopOpacity="0" />
            </linearGradient>
          </defs>
          <circle cx="1100" cy="380" r="320" fill="none" stroke="url(#arcGrad)" strokeWidth="1" />
          <circle cx="1100" cy="380" r="440" fill="none" stroke="url(#arcGrad)" strokeWidth="1" />
          <circle cx="1100" cy="380" r="560" fill="none" stroke="url(#arcGrad)" strokeWidth="1" />
        </svg>

        <div style={{ position: "relative", zIndex: 1, maxWidth: 1280, margin: "0 auto" }}>
          <div className="hero-grid" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.05fr)", gap: 64, alignItems: "center" }}>
          <div>
            <div style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                padding: "5px 12px 5px 8px", border: "1px solid var(--rule)", borderRadius: 999,
                background: "var(--card)", marginBottom: 28
              }}>
              <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--good)", boxShadow: "0 0 0 3px rgba(31,157,110,0.18)" }} />
              <span className="eyebrow" style={{ color: "var(--ink)" }}>{t("landing.eyebrow")}</span>
            </div>

            <h1 className="display" style={{ fontSize: 64, margin: "0 0 24px", color: "var(--ink)" }}>
              {t("landing.title.a")}<br />
              <span className="display-italic" style={{ color: "var(--ink-soft)" }}>{t("landing.title.b")}</span>
            </h1>

            <p style={{ fontSize: 17, lineHeight: 1.55, color: "var(--muted)", margin: "0 0 36px", maxWidth: 520 }}>
              {t("landing.lede")}
            </p>

            <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 40 }}>
              <Button kind="primary" size="lg" icon="arrow_forward" onClick={() => onNav("auth-signup")}>
                {t("landing.cta.try")}
              </Button>
              <Button kind="ghost" size="lg" icon="play_arrow" onClick={() => onNav("workspace")}>
                {t("landing.cta.demo")}
              </Button>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ display: "flex" }}>
                {["#3CC58F", "#5B8DEF", "#A06BD8", "#E2A33A"].map((c, i) =>
                  <div key={i} style={{
                    width: 28, height: 28, borderRadius: 999, background: c,
                    border: "2px solid var(--bg)", marginLeft: i ? -8 : 0
                  }} />
                  )}
              </div>
              <div style={{ fontSize: 13, color: "var(--muted)" }}>
                {t("landing.trust")} <span style={{ color: "var(--ink)", fontWeight: 600 }}></span>
              </div>
            </div>
          </div>

          {/* Hero visual — composite product preview */}
          <HeroVisual />
          </div>
        </div>
      </section>

      {/* HOW IT WORKS — 3 steps with timeline rule */}
      <section id="platform" style={{ padding: "72px 32px", borderTop: "1px solid var(--rule)" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <Eyebrow style={{ marginBottom: 12 }}>01 · {lang === "fr" ? "Méthode" : "Method"}</Eyebrow>
            <h2 className="display" style={{ fontSize: 42, margin: "0 0 12px" }}>{t("landing.steps.title")}</h2>
            <p style={{ color: "var(--muted)", fontSize: 16, margin: 0 }}>{t("landing.steps.lede")}</p>
          </div>

          <div style={{ position: "relative", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 32 }}>
            <div style={{ position: "absolute", top: 28, left: "16%", right: "16%", height: 1, background: "var(--rule)" }} />
            {[
            { n: "01", icon: "videocam", title: t("landing.step1.title"), body: t("landing.step1.body") },
            { n: "02", icon: "graph_3", title: t("landing.step2.title"), body: t("landing.step2.body") },
            { n: "03", icon: "auto_awesome", title: t("landing.step3.title"), body: t("landing.step3.body") }].
            map((s, i) =>
            <div key={i} style={{ position: "relative", textAlign: "center" }}>
                <div style={{
                width: 56, height: 56, borderRadius: 999, background: "var(--bg)",
                border: "1px solid var(--rule)", margin: "0 auto 24px",
                display: "inline-flex", alignItems: "center", justifyContent: "center",
                position: "relative", zIndex: 2
              }}>
                  <span className="icon" style={{ fontSize: 24, color: "var(--ink)" }}>{s.icon}</span>
                </div>
                <div className="mono" style={{ color: "var(--muted-2)", marginBottom: 8 }}>{s.n}</div>
                <h3 className="h3" style={{ margin: "0 0 10px" }}>{s.title}</h3>
                <p style={{ color: "var(--muted)", fontSize: 14, margin: 0, lineHeight: 1.55, maxWidth: 280, marginInline: "auto" }}>{s.body}</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* MULTIDIMENSIONAL — feature cards w/ density */}
      <section id="services" style={{ padding: "72px 32px", background: "var(--bg-2)", borderTop: "1px solid var(--rule)" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "end", marginBottom: 40, gap: 32, flexWrap: "wrap" }}>
            <div>
              <Eyebrow style={{ marginBottom: 12 }}>02 · {lang === "fr" ? "Capacités" : "Capabilities"}</Eyebrow>
              <h2 className="display" style={{ fontSize: 42, margin: "0 0 8px" }}>{t("landing.dim.title")}</h2>
              <p style={{ color: "var(--muted)", fontSize: 16, margin: 0, maxWidth: 540 }}>{t("landing.dim.lede")}</p>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <span className="mono" style={{ padding: "6px 10px", border: "1px solid var(--rule)", borderRadius: 999, background: "white", color: "var(--muted)" }}>14 {lang === "fr" ? "métriques" : "metrics"}</span>
              <span className="mono" style={{ padding: "6px 10px", border: "1px solid var(--rule)", borderRadius: 999, background: "white", color: "var(--muted)" }}>~2 min</span>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
            {[
            { icon: "graphic_eq", title: t("landing.dim.voice"), items: voiceItems, n: "01" },
            { icon: "accessibility_new", title: t("landing.dim.body"), items: bodyItems, n: "02" },
            { icon: "menu_book", title: t("landing.dim.content"), items: contentItems, n: "03" }].
            map((d, i) =>
            <Card key={i} hoverable padding={28} style={{ position: "relative" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
                  <div style={{
                  width: 44, height: 44, borderRadius: 12, background: "var(--bg)",
                  border: "1px solid var(--rule)", display: "inline-flex", alignItems: "center", justifyContent: "center"
                }}>
                    <span className="icon" style={{ fontSize: 22, color: "var(--ink)" }}>{d.icon}</span>
                  </div>
                  <span className="mono" style={{ color: "var(--muted-2)" }}>{d.n}</span>
                </div>
                <h3 className="h3" style={{ margin: "0 0 16px" }}>{d.title}</h3>
                <div className="hr" style={{ marginBottom: 16 }} />
                <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                  {d.items.map((it, j) =>
                <li key={j} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "10px 0", borderBottom: j < d.items.length - 1 ? "1px solid var(--rule-soft)" : "none",
                  fontSize: 13.5, color: "var(--ink-soft)"
                }}>
                      <span className="icon" style={{ fontSize: 16, color: "var(--muted-2)" }}>check_small</span>
                      {it}
                    </li>
                )}
                </ul>
              </Card>
            )}
          </div>
        </div>
      </section>

      {/* PRICING */}
      <section id="pricing" style={{ padding: "72px 32px", borderTop: "1px solid var(--rule)" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 48 }}>
            <Eyebrow style={{ marginBottom: 12 }}>03 · {lang === "fr" ? "Tarification" : "Pricing"}</Eyebrow>
            <h2 className="display" style={{ fontSize: 42, margin: "0 0 12px" }}>{t("landing.pricing.title")}</h2>
            <p style={{ color: "var(--muted)", fontSize: 16, margin: 0 }}>{t("landing.pricing.lede")}</p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Free */}
            <Card padding={32}>
              <Eyebrow>{t("landing.tier.free")}</Eyebrow>
              <div className="display" style={{ fontSize: 36, margin: "12px 0 28px" }}>{t("landing.tier.free.price")}</div>
              <div className="hr" style={{ marginBottom: 20 }} />
              <ul style={{ listStyle: "none", padding: 0, margin: "0 0 32px" }}>
                {STRINGS["landing.tier.free.items"][lang].map((it, i) =>
                <li key={i} style={{ display: "flex", gap: 10, padding: "8px 0", fontSize: 14, color: "var(--ink-soft)" }}>
                    <span className="icon" style={{ fontSize: 18, color: "var(--good)" }}>check_circle</span>
                    {it}
                  </li>
                )}
              </ul>
              <Button kind="ghost" size="lg" style={{ width: "100%" }} onClick={() => onNav("auth-signup")}>{t("landing.tier.cta.start")}</Button>
            </Card>

            {/* Pro */}
            <Card padding={32} style={{ background: "var(--ink)", border: "1px solid var(--ink)", color: "white", position: "relative" }}>
              <div style={{
                position: "absolute", top: 16, right: 16, padding: "4px 10px",
                borderRadius: 999, background: "rgba(255,255,255,0.14)",
                fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase"
              }}>{t("landing.tier.popular")}</div>
              <div className="eyebrow" style={{ color: "rgba(255,255,255,0.6)" }}>{t("landing.tier.pro")}</div>
              <div className="display" style={{ fontSize: 36, margin: "12px 0 28px" }}>
                {t("landing.tier.pro.price")}
                <span style={{ fontSize: 14, fontWeight: 400, color: "rgba(255,255,255,0.6)", marginLeft: 6 }}>{t("landing.tier.pro.per")}</span>
              </div>
              <div style={{ height: 1, background: "rgba(255,255,255,0.12)", marginBottom: 20 }} />
              <ul style={{ listStyle: "none", padding: 0, margin: "0 0 32px" }}>
                {STRINGS["landing.tier.pro.items"][lang].map((it, i) =>
                <li key={i} style={{ display: "flex", gap: 10, padding: "8px 0", fontSize: 14, color: "rgba(255,255,255,0.88)" }}>
                    <span className="icon" style={{ fontSize: 18, color: "white" }}>check_circle</span>
                    {it}
                  </li>
                )}
              </ul>
              <Button kind="invert" size="lg" style={{ width: "100%" }} onClick={() => onNav("auth-signup")}>{t("landing.tier.cta.pro")}</Button>
            </Card>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section id="about" style={{ padding: "72px 32px", borderTop: "1px solid var(--rule)", background: "var(--bg-2)" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", display: "grid", gridTemplateColumns: "0.6fr 1fr", gap: 64 }}>
          <div>
            <Eyebrow style={{ marginBottom: 12 }}>04 · FAQ</Eyebrow>
            <h2 className="display" style={{ fontSize: 36, margin: 0 }}>{t("landing.faq.title")}</h2>
          </div>
          <div>
            {[1, 2, 3, 4].map((n) => <FaqItem key={n} q={t(`landing.faq.q${n}`)} a={t(`landing.faq.a${n}`)} />)}
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer style={{ padding: "56px 32px 32px", borderTop: "1px solid var(--rule)" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr", gap: 48 }}>
          <div>
            <Logo />
            <p style={{ color: "var(--muted)", fontSize: 13, lineHeight: 1.6, margin: "16px 0 0", maxWidth: 320 }}>
              {t("landing.footer.tagline")}
            </p>
          </div>
          {[
          { title: t("landing.footer.legal"), items: [t("landing.footer.privacy"), t("landing.footer.terms")] },
          { title: t("landing.footer.company"), items: [t("landing.footer.contact"), t("landing.footer.careers"), t("nav.about")] },
          { title: t("landing.footer.social"), items: ["LinkedIn", "Twitter / X", "GitHub"] }].
          map((col, i) =>
          <div key={i}>
              <Eyebrow style={{ marginBottom: 12 }}>{col.title}</Eyebrow>
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {col.items.map((it, j) =>
              <li key={j} style={{ padding: "5px 0" }}>
                    <a href="#" style={{ color: "var(--ink-soft)", textDecoration: "none", fontSize: 13.5 }}>{it}</a>
                  </li>
              )}
              </ul>
            </div>
          )}
        </div>
        <div className="hr" style={{ margin: "40px 0 16px" }} />
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", justifyContent: "space-between", color: "var(--muted)", fontSize: 12 }}>
          <span>{t("landing.footer.copy")}</span>
          <span className="mono">v1.2.0 · build 2603</span>
        </div>
      </footer>
    </div>);

}

function FaqItem({ q, a }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: "1px solid var(--rule)" }}>
      <button onClick={() => setOpen((o) => !o)} style={{
        width: "100%", textAlign: "left", padding: "20px 0",
        display: "flex", justifyContent: "space-between", alignItems: "center", gap: 24,
        background: "transparent", border: 0, cursor: "pointer", color: "var(--ink)",
        fontFamily: "var(--display)", fontWeight: 600, fontSize: 17
      }}>
        {q}
        <span className="icon" style={{ fontSize: 22, transform: open ? "rotate(45deg)" : "none", transition: "transform 0.2s" }}>add</span>
      </button>
      {open && <div style={{ padding: "0 0 20px", color: "var(--muted)", fontSize: 14.5, lineHeight: 1.6, maxWidth: 580 }}>{a}</div>}
    </div>);

}

// Hero visual — soft lavender bloom + browser chrome embedding real product screenshot
function HeroVisual() {
  return (
    <div style={{ position: "relative", width: "100%", minWidth: 320 }}>
      {/* Browser chrome card */}
      <div style={{
        position: "relative",
        background: "white",
        borderRadius: 14,
        overflow: "hidden",
        boxShadow: "0 40px 80px -28px rgba(80, 60, 180, 0.28), 0 12px 28px -10px rgba(15, 8, 102, 0.14)",
        border: "1px solid rgba(15, 8, 102, 0.06)",
        zIndex: 2
      }}>
        {/* Top bar */}
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "11px 14px",
          borderBottom: "1px solid #f0eef7",
          background: "#fbfbff"
        }}>
          <div style={{ display: "flex", gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 999, background: "#FF5F57" }} />
            <span style={{ width: 10, height: 10, borderRadius: 999, background: "#FEBC2E" }} />
            <span style={{ width: 10, height: 10, borderRadius: 999, background: "#28C840" }} />
          </div>
          <div style={{
            flex: 1,
            margin: "0 8px",
            background: "white",
            border: "1px solid #ece9f5",
            borderRadius: 6,
            padding: "4px 10px",
            display: "flex", alignItems: "center", gap: 6,
            fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)"
          }}>
            <span className="icon" style={{ fontSize: 12, color: "var(--muted-2)" }}>lock</span>
            app.nood.ai/sessions
          </div>
          <span className="icon" style={{ fontSize: 16, color: "var(--muted-2)" }}>more_horiz</span>
        </div>
        {/* Screenshot */}
        <img
          src="assets/hero-history.png"
          alt="NOOD app preview"
          style={{ display: "block", width: "100%", height: "auto" }} />
      </div>

      {/* Floating mini-card — bottom-left, peeking out, low opacity to feel layered */}
      <div style={{
        position: "absolute", bottom: -28, left: -32,
        background: "white",
        borderRadius: 12,
        padding: "12px 14px",
        boxShadow: "0 20px 40px -16px rgba(80, 60, 180, 0.30)",
        border: "1px solid rgba(15, 8, 102, 0.06)",
        zIndex: 3,
        display: "flex", alignItems: "center", gap: 12
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: "linear-gradient(140deg, #6b5cff 0%, #a06bd8 100%)",
          display: "inline-flex", alignItems: "center", justifyContent: "center"
        }}>
          <span className="icon" style={{ fontSize: 18, color: "white" }}>auto_awesome</span>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.08em" }}>Score · 30j</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span className="num" style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.01em", color: "var(--ink)" }}>+14.5</span>
            <span style={{ fontSize: 11, color: "var(--good)", fontWeight: 600 }}>↑ trending</span>
          </div>
        </div>
      </div>

      {/* Floating mini-card — top-right, score */}
      <div style={{
        position: "absolute", top: -22, right: -22,
        background: "white",
        borderRadius: 12,
        padding: "10px 14px",
        boxShadow: "0 20px 40px -16px rgba(80, 60, 180, 0.30)",
        border: "1px solid rgba(15, 8, 102, 0.06)",
        zIndex: 3
      }}>
        <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 2 }}>Pitch — Demo Day</div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
          <span className="num" style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.01em", color: "var(--ink)" }}>72.5</span>
          <span style={{ fontSize: 11, color: "var(--muted)", fontWeight: 500 }}>· B</span>
        </div>
      </div>
    </div>);

}

window.Landing = Landing;