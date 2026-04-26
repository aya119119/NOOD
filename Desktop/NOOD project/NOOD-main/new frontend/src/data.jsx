// Mock data — based on the real report.json shape but enriched for UI demo
const REPORT = {
  meta: {
    name: { fr: "Pitch produit — Demo Day", en: "Product pitch — Demo Day" },
    date: "2026-04-22T14:32:00",
    duration_s: 187,
    file: "demo_day_pitch_v3.mp4",
    fileSize: "47.2 MB",
  },
  overall: { score: 72.5, grade: "B", proficiency: "Strong" },
  prev: { score: 68.5, grade: "C+" },
  avg30: 70.2,
  components: {
    voice: 76,
    body: 64,
    tone: 78,
    content: 70,
  },
  body_language: {
    dominant: "Happy",
    dominant_pct: 38.2,
    frames_analyzed: 760,
    distribution: [
      { label: "Happy",     pct: 38.2, color: "#3CC58F" },
      { label: "Excited",   pct: 18.5, color: "#5B8DEF" },
      { label: "Confused",  pct: 15.3, color: "#E2A33A" },
      { label: "Tension",   pct: 12.1, color: "#A06BD8" },
      { label: "Surprised", pct:  8.9, color: "#E07AB0" },
      { label: "Sad",       pct:  4.2, color: "#7B7494" },
      { label: "Angry",     pct:  1.8, color: "#C94B4B" },
      { label: "Pain",      pct:  1.0, color: "#D17B3F" },
    ],
    interpretation: {
      fr: "Distribution équilibrée avec une dominante positive. Le pic de tension (12%) se concentre autour des chiffres techniques.",
      en: "Balanced distribution with a positive lead. The tension spike (12%) clusters around technical numbers."
    },
    timeline: Array.from({ length: 60 }, (_, i) => ({
      t: i * (187/60),
      emotion: ["Happy","Excited","Confused","Tension","Happy","Surprised"][Math.floor(Math.sin(i*0.7)*3 + 3)] || "Happy"
    })),
  },
  speech: {
    grade: "B",
    metrics: {
      wpm:        { score: 0.55, raw: 148.3, unit: "wpm",  ideal: [120, 170], range: [80, 220], label: { fr: "Débit de parole", en: "Speaking pace" }, feedback: { fr: "Légèrement rapide. Idéal entre 120-170 mots/min pour ce contexte.", en: "Slightly fast. Ideal range 120-170 wpm for this context." } },
      fillers:    { score: 0.78, raw: 2.4,   unit: "%",    ideal: [0, 3],     range: [0, 12],   label: { fr: "Mots parasites", en: "Filler words" },  feedback: { fr: "Très bonne maîtrise (2.4% de mots parasites). Continuez ainsi.", en: "Excellent control (2.4% fillers). Keep it up." } },
      pitch:      { score: 0.32, raw: 24.6,  unit: "Hz σ", ideal: [35, 65],   range: [10, 90],  label: { fr: "Variation de hauteur", en: "Pitch variation" }, feedback: { fr: "Tendance monotone. Variez davantage pour souligner les points clés.", en: "Tending monotone. Vary more to emphasize key points." } },
      energy:     { score: 0.58, raw: 0.022, unit: "RMS σ",ideal: [0.025, 0.04], range: [0, 0.06], label: { fr: "Variation d'énergie", en: "Energy variation" }, feedback: { fr: "Bonne dynamique, légèrement plate sur la fin.", en: "Good dynamics, slightly flat toward the end." } },
      pause:      { score: 0.72, raw: 14.2,  unit: "%",    ideal: [10, 20],   range: [0, 40],   label: { fr: "Ratio de pauses", en: "Pause ratio" },     feedback: { fr: "Bon usage des pauses. Livraison confiante et mesurée.", en: "Good use of pauses. Confident, measured delivery." } },
      emotion:    { score: 0.85, raw: "happy", unit: "",   label: { fr: "Émotion vocale", en: "Vocal emotion" },         feedback: { fr: "Ton vocal engagé et positif — excellent pour la connexion avec l'audience.", en: "Engaged, positive tone — excellent for audience connection." } },
    },
    transcript_preview: {
      fr: "Bonjour à tous. Aujourd'hui je vais vous présenter NOOD, une plateforme qui transforme la manière dont nous nous entraînons à parler en public. Le problème est simple : la plupart d'entre nous improvisent leurs présentations sans aucun retour structuré…",
      en: "Hello everyone. Today I'm going to present NOOD, a platform that transforms how we train ourselves to speak in public. The problem is simple: most of us improvise our presentations without any structured feedback…"
    },
  },
  tone: {
    topic: { fr: "Présentation d'un nouveau produit IA pour l'éducation", en: "Pitch of a new AI product for education" },
    context: { fr: "Demo Day · jury technique et investisseurs", en: "Demo Day · technical jury & investors" },
    fit: 0.78,
    fit_label: { fr: "Approprié", en: "Appropriate" },
    mismatches: [
      {
        severity: "low",
        moment_s: 92,
        moment_label: { fr: "Milieu de la présentation", en: "Mid-presentation" },
        observed: { fr: "Légèrement monotone par moments", en: "Slightly monotone at moments" },
        expected: { fr: "Plus d'enthousiasme pour les points d'action", en: "More enthusiasm for action points" },
        reason: { fr: "L'énergie baisse lors de la présentation des données chiffrées, alors qu'un ton plus haut maintiendrait l'attention.", en: "Energy drops during numerical sections, where a higher pitch would hold attention." }
      },
      {
        severity: "medium",
        moment_s: 158,
        moment_label: { fr: "Conclusion", en: "Closing" },
        observed: { fr: "Débit accéléré, ton aplati", en: "Faster pace, flat tone" },
        expected: { fr: "Pause finale, conclusion ancrée", en: "Final pause, anchored close" },
        reason: { fr: "La conclusion donne l'impression d'être pressée — l'auditoire perd la phrase clé.", en: "The closing feels rushed — the audience loses the key sentence." }
      },
    ],
    coaching_tips: [
      {
        impact: "high",
        moment_s: 92,
        title: { fr: "Variez votre énergie sur les chiffres", en: "Vary your energy on numbers" },
        body: { fr: "Lorsque vous présentez des résultats chiffrés, augmentez votre volume et marquez une micro-pause après chaque chiffre clé.", en: "When presenting numerical results, raise volume and add a micro-pause after each key figure." }
      },
      {
        impact: "high",
        moment_s: 158,
        title: { fr: "Pause finale, deux secondes", en: "Final pause, two seconds" },
        body: { fr: "Avant votre dernière phrase, faites une pause franche de deux secondes. C'est ce qui ancre votre message.", en: "Before your final sentence, take a clean two-second pause. That's what anchors your message." }
      },
      {
        impact: "medium",
        moment_s: 45,
        title: { fr: "Question rhétorique en ouverture", en: "Open with a rhetorical question" },
        body: { fr: "Démarrez chaque section par une question pour engager l'audience plutôt qu'une affirmation.", en: "Open each section with a question to engage the audience instead of a statement." }
      },
      {
        impact: "low",
        moment_s: 0,
        title: { fr: "Résumé d'une phrase par section", en: "One-sentence summary per section" },
        body: { fr: "Terminez chaque section par une phrase qui résume l'idée. Aide la mémoire de l'auditoire.", en: "End each section with a one-sentence summary. Helps audience recall." }
      },
    ]
  },
  language: {
    grammar: 8.4,
    structure: 7.2,
    vocab: 8.9,
    fluency: 9.0,
    overall: 8.4,
  },
};

// Past sessions for History
const HISTORY = [
  { id: "s7", date: "2026-04-22", name: { fr: "Pitch produit — Demo Day", en: "Product pitch — Demo Day" }, duration: 187, score: 72.5, grade: "B", kind: { fr: "Pitch", en: "Pitch" } },
  { id: "s6", date: "2026-04-15", name: { fr: "Soutenance PFE — répétition 2", en: "Thesis defense — rehearsal 2" }, duration: 612, score: 68.5, grade: "C+", kind: { fr: "Soutenance", en: "Defense" } },
  { id: "s5", date: "2026-04-08", name: { fr: "Présentation client", en: "Client presentation" }, duration: 421, score: 71.0, grade: "B", kind: { fr: "Pro", en: "Pro" } },
  { id: "s4", date: "2026-03-31", name: { fr: "Soutenance PFE — répétition 1", en: "Thesis defense — rehearsal 1" }, duration: 608, score: 64.0, grade: "C", kind: { fr: "Soutenance", en: "Defense" } },
  { id: "s3", date: "2026-03-22", name: { fr: "Hackathon — pitch final", en: "Hackathon — final pitch" }, duration: 122, score: 76.5, grade: "B+", kind: { fr: "Pitch", en: "Pitch" } },
  { id: "s2", date: "2026-03-12", name: { fr: "Présentation cours IA", en: "AI course presentation" }, duration: 540, score: 65.5, grade: "C", kind: { fr: "Cours", en: "Class" } },
  { id: "s1", date: "2026-03-02", name: { fr: "Première analyse", en: "First analysis" }, duration: 200, score: 58.0, grade: "D+", kind: { fr: "Pratique", en: "Practice" } },
];

Object.assign(window, { REPORT, HISTORY });
