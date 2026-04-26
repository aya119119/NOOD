// Shared UI components — header, language toggle, dev pill toolbar, basic primitives

const { useState, useEffect, useRef, useContext, useCallback, useMemo } = React;

// ─── Logo ──────────────────────────────────────────────────────────────
function Logo({ size = 26, onClick }) {
  // logo image is 563×345 wide-format and already contains the "NOOD" wordmark
  const h = size;
  const w = h * (563 / 345);
  return (
    <div onClick={onClick} style={{ display: "inline-flex", alignItems: "center", cursor: onClick ? "pointer" : "default", userSelect: "none" }}>
      <img src="assets/nood_logo.png" alt="NOOD" height={h} style={{ height: h, width: w, display: "block", objectFit: "contain" }} />
    </div>
  );
}

// ─── Language toggle ──────────────────────────────────────────────────
function LangToggle() {
  const { lang, setLang } = useT();
  return (
    <div style={{ display: "inline-flex", border: "1px solid var(--rule)", borderRadius: 999, padding: 2, background: "var(--card)" }}>
      {["fr", "en"].map((l) =>
      <button key={l}
      onClick={() => setLang(l)}
      style={{
        border: 0, background: lang === l ? "var(--ink)" : "transparent",
        color: lang === l ? "white" : "var(--muted)",
        fontFamily: "var(--body)", fontSize: 11, fontWeight: 600,
        padding: "5px 11px", borderRadius: 999, cursor: "pointer",
        textTransform: "uppercase", letterSpacing: "0.08em", transition: "all 0.15s"
      }}>
          {l === "en" ? "FR" : l}
        </button>
      )}
    </div>);

}

// ─── Floating dev/edit pill (kept per user request) ───────────────────
function DevToolPill() {
  const icons = ["chat_add_on", "lock", "content_copy", "delete", "more_horiz"];
  return (
    <div style={{
      position: "fixed", top: 18, left: "50%", transform: "translateX(-50%)",
      display: "flex", gap: 4, padding: 6,
      background: "rgba(255,255,255,0.92)", backdropFilter: "blur(12px)",
      border: "1px solid var(--rule)", borderRadius: 999,
      boxShadow: "0 6px 24px -8px rgba(15,8,102,0.14)", zIndex: 80
    }}>
      {icons.map((ic, i) =>
      <button key={i} style={{
        width: 32, height: 32, border: 0, background: "transparent",
        borderRadius: 999, cursor: "pointer", color: "var(--ink)",
        display: "inline-flex", alignItems: "center", justifyContent: "center"
      }}
      onMouseEnter={(e) => e.currentTarget.style.background = "var(--hover)"}
      onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}>
        
          <span className="icon" style={{ fontSize: 18 }}>{ic}</span>
        </button>
      )}
    </div>);

}

// ─── Header ────────────────────────────────────────────────────────────
function Header({ variant = "marketing", onNav, route, signedIn }) {
  const { t } = useT();
  const isApp = variant === "app";

  return (
    <header style={{
      position: "sticky", top: 0, zIndex: 60,
      background: "rgba(251,251,255,0.85)", backdropFilter: "blur(14px)",
      borderBottom: "1px solid var(--rule)"
    }}>
      <div style={{
        maxWidth: 1280, margin: "0 auto", padding: "14px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between", gap: 24
      }}>
        <Logo onClick={() => onNav(signedIn ? "history" : "landing")} />

        {!isApp ?
        <nav style={{ display: "flex", gap: 28, alignItems: "center" }}>
            {[
          { id: "platform", label: t("nav.platform") },
          { id: "services", label: t("nav.services") },
          { id: "pricing", label: t("nav.pricing") },
          { id: "about", label: t("nav.about") }].
          map((item) =>
          <a key={item.id} href={`#${item.id}`} style={{
            color: "var(--ink)", textDecoration: "none", fontSize: 14, fontWeight: 500,
            padding: "6px 0", borderBottom: "1.5px solid transparent",
            transition: "border-color 0.15s"
          }}
          onMouseEnter={(e) => e.currentTarget.style.borderColor = "var(--ink)"}
          onMouseLeave={(e) => e.currentTarget.style.borderColor = "transparent"}>
            {item.label}</a>
          )}
          </nav> :

        <nav style={{ display: "flex", gap: 4, alignItems: "center" }}>
            {[
          { id: "workspace", icon: "add_circle", label: { fr: "Nouvelle", en: "New" } },
          { id: "history", icon: "history", label: { fr: "Historique", en: "History" } }].
          map((item) => {
            const active = route === item.id || item.id === "workspace" && route === "processing";
            return (
              <button key={item.id} onClick={() => onNav(item.id)} style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                padding: "8px 14px", borderRadius: 999,
                background: active ? "var(--ink)" : "transparent",
                color: active ? "white" : "var(--ink)",
                border: 0, fontSize: 13, fontWeight: 500, cursor: "pointer",
                fontFamily: "var(--body)", transition: "all 0.15s"
              }}>
                  <span className="icon" style={{ fontSize: 18 }}>{item.icon}</span>
                  <NavLabel item={item} />
                </button>);

          })}
          </nav>
        }

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <LangToggle />
          {!signedIn ?
          <>
              <button onClick={() => onNav("auth-signin")} style={{
              border: "1px solid var(--rule)", background: "white", color: "var(--ink)",
              padding: "8px 16px", borderRadius: 999, fontWeight: 500, fontSize: 13, cursor: "pointer"
            }}>{t("nav.signin")}</button>
              <button onClick={() => onNav("auth-signup")} style={{
              border: 0, background: "var(--ink)", color: "white",
              padding: "9px 18px", borderRadius: 999, fontWeight: 500, fontSize: 13, cursor: "pointer"
            }}>{t("nav.signup")}</button>
            </> :

          <UserMenu onNav={onNav} />
          }
        </div>
      </div>
    </header>);

}

function NavLabel({ item }) {
  const { lang } = useT();
  return <span>{item.label[lang]}</span>;
}

function UserMenu({ onNav }) {
  const [open, setOpen] = useState(false);
  const { t } = useT();
  return (
    <div style={{ position: "relative" }}>
      <button onClick={() => setOpen((o) => !o)} style={{
        width: 36, height: 36, borderRadius: 999, border: "1px solid var(--rule)",
        background: "linear-gradient(135deg, #6b5cff 0%, #0F0866 100%)",
        color: "white", fontWeight: 600, fontSize: 13, cursor: "pointer",
        fontFamily: "var(--display)"
      }}>SA</button>
      {open &&
      <div style={{
        position: "absolute", right: 0, top: 44, minWidth: 220,
        background: "white", border: "1px solid var(--rule)", borderRadius: 14,
        boxShadow: "0 12px 40px -12px rgba(15,8,102,0.18)",
        padding: 6, zIndex: 100
      }}>
          <div style={{ padding: "10px 12px 8px" }}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Sara Amrani</div>
            <div style={{ color: "var(--muted)", fontSize: 12 }}>sara@enseirb.fr</div>
          </div>
          <div className="hr" style={{ margin: "6px 0" }} />
          {[
        { label: t("nav.history"), icon: "history", id: "history" },
        { label: t("nav.account"), icon: "person", id: "account" },
        { label: t("nav.signout"), icon: "logout", id: "landing" }].
        map((it) =>
        <button key={it.id} onClick={() => {setOpen(false);onNav(it.id);}} style={{
          display: "flex", alignItems: "center", gap: 10, width: "100%",
          padding: "9px 12px", border: 0, background: "transparent",
          borderRadius: 8, cursor: "pointer", color: "var(--ink)",
          fontFamily: "var(--body)", fontSize: 13, textAlign: "left"
        }}
        onMouseEnter={(e) => e.currentTarget.style.background = "var(--hover)"}
        onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}>
          
              <span className="icon" style={{ fontSize: 18, color: "var(--muted)" }}>{it.icon}</span>
              {it.label}
            </button>
        )}
        </div>
      }
    </div>);

}

// ─── Card primitive ───────────────────────────────────────────────────
function Card({ children, style, padding = 24, hoverable, onClick }) {
  return (
    <div onClick={onClick} style={{
      background: "var(--card)", border: "1px solid var(--rule)", borderRadius: 14,
      padding, transition: "all 0.18s",
      cursor: onClick ? "pointer" : "default",
      ...style
    }}
    onMouseEnter={hoverable ? (e) => e.currentTarget.style.borderColor = "var(--muted-2)" : undefined}
    onMouseLeave={hoverable ? (e) => e.currentTarget.style.borderColor = "var(--rule)" : undefined}>
      {children}</div>);

}

function Button({ children, kind = "primary", size = "md", icon, onClick, type = "button", style, disabled }) {
  const sizes = {
    sm: { padding: "6px 12px", fontSize: 12, height: 30 },
    md: { padding: "10px 18px", fontSize: 13, height: 40 },
    lg: { padding: "14px 24px", fontSize: 15, height: 50 }
  };
  const kinds = {
    primary: { background: "var(--ink)", color: "white", border: "1px solid var(--ink)" },
    ghost: { background: "transparent", color: "var(--ink)", border: "1px solid var(--rule)" },
    quiet: { background: "transparent", color: "var(--ink)", border: "1px solid transparent" },
    invert: { background: "white", color: "var(--ink)", border: "1px solid white" }
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled} style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
      borderRadius: 999, fontFamily: "var(--body)", fontWeight: 500, cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.5 : 1, transition: "transform 0.06s, box-shadow 0.18s",
      ...sizes[size], ...kinds[kind], ...style
    }}
    onMouseDown={(e) => !disabled && (e.currentTarget.style.transform = "translateY(1px)")}
    onMouseUp={(e) => e.currentTarget.style.transform = ""}
    onMouseLeave={(e) => e.currentTarget.style.transform = ""}>
      
      {icon && <span className="icon" style={{ fontSize: 18 }}>{icon}</span>}
      {children}
    </button>);

}

// ─── Eyebrow / labels ────────────────────────────────────────────────
function Eyebrow({ children, style }) {
  return <div className="eyebrow" style={style}>{children}</div>;
}

// ─── Sparkline ────────────────────────────────────────────────────────
function Sparkline({ values, width = 120, height = 32, color = "var(--ink)" }) {
  const min = Math.min(...values),max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = i / (values.length - 1) * width;
    const y = height - (v - min) / range * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={width} cy={height - (values[values.length - 1] - min) / range * (height - 4) - 2} r="2.5" fill={color} />
    </svg>);

}

// ─── Bar Range (with ideal zone) ─────────────────────────────────────
function MetricBar({ value, range = [0, 100], ideal, size = "md" }) {
  const [min, max] = range;
  const pct = Math.max(0, Math.min(100, (value - min) / (max - min) * 100));
  const idealStart = ideal ? (ideal[0] - min) / (max - min) * 100 : 0;
  const idealEnd = ideal ? (ideal[1] - min) / (max - min) * 100 : 0;
  const h = size === "sm" ? 6 : 8;
  return (
    <div style={{ position: "relative", height: h, background: "var(--rule-soft)", borderRadius: h }}>
      {ideal &&
      <div style={{
        position: "absolute", left: `${idealStart}%`, width: `${idealEnd - idealStart}%`,
        top: 0, bottom: 0, background: "rgba(31,157,110,0.18)", borderRadius: h
      }} />
      }
      <div style={{
        position: "absolute", left: `calc(${pct}% - 1px)`, top: -2, bottom: -2,
        width: 2, background: "var(--ink)", borderRadius: 2
      }} />
    </div>);

}

// ─── Stacked emotion bar ──────────────────────────────────────────────
function StackedBar({ items, height = 10 }) {
  return (
    <div style={{ display: "flex", height, borderRadius: 999, overflow: "hidden", border: "1px solid var(--rule)" }}>
      {items.map((it, i) =>
      <div key={i} title={`${it.label}: ${it.pct.toFixed(1)}%`}
      style={{ width: `${it.pct}%`, background: it.color, transition: "filter 0.15s" }} />
      )}
    </div>);

}

// ─── Score gauge (radial-ish, but simple: number + arc) ──────────────
function ScoreRing({ value, size = 140, stroke = 6, color = "var(--ink)" }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - value / 100 * c;
  return (
    <svg width={size} height={size} style={{ display: "block" }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--rule)" strokeWidth={stroke} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
      strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
      transform={`rotate(-90 ${size / 2} ${size / 2})`}
      style={{ transition: "stroke-dashoffset 0.6s" }} />
    </svg>);

}

// ─── Animated waveform ────────────────────────────────────────────────
function MiniWaveform({ active = true, bars = 18, color = "var(--ink)" }) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 2, height: 18 }}>
      {Array.from({ length: bars }).map((_, i) =>
      <div key={i} style={{
        width: 2, height: 18, background: color, borderRadius: 2,
        transformOrigin: "center",
        animation: active ? `wave 0.9s ease-in-out ${i * 0.05}s infinite` : "none",
        opacity: active ? 1 : 0.4
      }} />
      )}
    </div>);

}

Object.assign(window, { Logo, LangToggle, DevToolPill, Header, Card, Button, Eyebrow, Sparkline, MetricBar, StackedBar, ScoreRing, MiniWaveform });