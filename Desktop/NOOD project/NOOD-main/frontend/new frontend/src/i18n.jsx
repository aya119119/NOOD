// Bilingual strings — French primary, English secondary
const STRINGS = {
  // Header / common
  "nav.platform":     { fr: "Plateforme",  en: "Platform" },
  "nav.services":     { fr: "Services",    en: "Services" },
  "nav.pricing":      { fr: "Tarifs",      en: "Pricing" },
  "nav.about":        { fr: "À propos",    en: "About" },
  "nav.signin":       { fr: "Se connecter", en: "Sign in" },
  "nav.signup":       { fr: "Inscription gratuite", en: "Get started" },
  "nav.getstarted":   { fr: "Commencer",   en: "Get started" },
  "nav.dashboard":    { fr: "Tableau de bord", en: "Dashboard" },
  "nav.history":      { fr: "Historique",  en: "History" },
  "nav.account":      { fr: "Compte",      en: "Account" },
  "nav.signout":      { fr: "Déconnexion", en: "Sign out" },

  // Landing
  "landing.eyebrow":  { fr: "Nouveau · Analyse IA en temps réel", en: "New · Real-time AI analysis" },
  "landing.title.a":  { fr: "Transformez votre communication en", en: "Turn your communication into" },
  "landing.title.b":  { fr: "performance réelle.", en: "real performance." },
  "landing.lede":     { fr: "Le coach IA qui décompose votre voix, votre langage corporel et la cohérence de votre discours — en moins de cinq minutes.", en: "The AI coach that breaks down your voice, body language, and message coherence — in under five minutes." },
  "landing.cta.try":  { fr: "Essayer gratuitement", en: "Try it free" },
  "landing.cta.demo": { fr: "Voir une démo", en: "See a demo" },
  "landing.trust":    { fr: "Utilisé par des étudiants et des équipes en", en: "Used by students and teams across" },

  "landing.steps.title": { fr: "Un parcours vers l'excellence oratoire", en: "A path to speaking mastery" },
  "landing.steps.lede":  { fr: "Trois étapes, un seul rapport, des progrès mesurables.", en: "Three steps, one report, measurable progress." },
  "landing.step1.title": { fr: "Enregistrez ou importez", en: "Record or import" },
  "landing.step1.body":  { fr: "Filmez-vous depuis l'application, ou déposez un fichier MP4, MOV, WEBM (jusqu'à 500 Mo).", en: "Record from the app, or drop in MP4, MOV, WEBM (up to 500 MB)." },
  "landing.step2.title": { fr: "L'IA analyse vos signaux", en: "AI reads your signals" },
  "landing.step2.body":  { fr: "Voix, posture, gestes, cohérence du discours. 8 émotions corporelles détectées par seconde.", en: "Voice, posture, gestures, message fit. 8 body emotions detected per second." },
  "landing.step3.title": { fr: "Recevez vos conseils", en: "Get coaching" },
  "landing.step3.body":  { fr: "Un rapport haute fidélité avec des recommandations classées par impact, ancrées au moment exact de la vidéo.", en: "A high-fidelity report with impact-ranked recommendations, pinned to the exact video moment." },

  "landing.dim.title":   { fr: "Analyse multidimensionnelle", en: "Multidimensional analysis" },
  "landing.dim.lede":    { fr: "Nous décomposons chaque nuance de votre intervention.", en: "We break down every nuance of your delivery." },
  "landing.dim.voice":   { fr: "Voix", en: "Voice" },
  "landing.dim.body":    { fr: "Corps", en: "Body" },
  "landing.dim.content": { fr: "Contenu", en: "Content" },
  "landing.dim.voice.items":   { fr: ["Débit (mots/min)", "Mots parasites", "Variations de hauteur", "Pauses stratégiques", "Émotion vocale"], en: ["Pace (words/min)", "Filler words", "Pitch variation", "Strategic pauses", "Vocal emotion"] },
  "landing.dim.body.items":    { fr: ["Posture & équilibre", "Gestuelle illustrative", "Contact visuel", "Occupation de l'espace", "Distribution émotionnelle"], en: ["Posture & balance", "Illustrative gestures", "Eye contact", "Use of space", "Emotion distribution"] },
  "landing.dim.content.items": { fr: ["Cohérence du ton", "Structure narrative", "Richesse lexicale", "Force d'introduction", "Adéquation au contexte"], en: ["Tone coherence", "Narrative structure", "Lexical richness", "Opening strength", "Context fit"] },

  "landing.pricing.title": { fr: "Investissez dans votre avenir", en: "Invest in your future" },
  "landing.pricing.lede":  { fr: "Commencez gratuitement. Passez Pro quand vous êtes prêt.", en: "Start free. Go Pro when you're ready." },
  "landing.tier.free":     { fr: "Découverte", en: "Free" },
  "landing.tier.free.price": { fr: "Gratuit", en: "Free" },
  "landing.tier.free.items": {
    fr: ["3 analyses par mois", "Rapport vocal complet", "Historique 7 jours", "Analyse corporelle limitée"],
    en: ["3 analyses per month", "Full vocal report", "7-day history", "Limited body analysis"]
  },
  "landing.tier.pro":     { fr: "Premium Pro", en: "Premium Pro" },
  "landing.tier.pro.price": { fr: "29 MAD", en: "29 MAD" },
  "landing.tier.pro.per":   { fr: "/ mois", en: "/ month" },
  "landing.tier.pro.items": {
    fr: ["Analyses illimitées", "IA complète (Voix + Corps + Ton)", "Historique illimité & tendances", "Coaching personnalisé", "Export PDF & lien partage"],
    en: ["Unlimited analyses", "Full AI (Voice + Body + Tone)", "Unlimited history & trends", "Personalized coaching", "PDF export & share link"]
  },
  "landing.tier.popular": { fr: "Plus populaire", en: "Most popular" },
  "landing.tier.cta.start": { fr: "Commencer", en: "Get started" },
  "landing.tier.cta.pro":   { fr: "Devenir Premium", en: "Go Premium" },

  "landing.faq.title":  { fr: "Questions fréquentes", en: "Frequently asked" },
  "landing.faq.q1":     { fr: "Mes vidéos sont-elles privées ?", en: "Are my videos private?" },
  "landing.faq.a1":     { fr: "Oui. Vos enregistrements sont chiffrés en transit et au repos. Vous pouvez les supprimer à tout moment depuis votre historique.", en: "Yes. Your recordings are encrypted in transit and at rest. You can delete them anytime from your history." },
  "landing.faq.q2":     { fr: "Quelle est la durée maximale ?", en: "What's the max length?" },
  "landing.faq.a2":     { fr: "500 Mo par fichier — environ 30 minutes de vidéo en qualité standard.", en: "500 MB per file — roughly 30 minutes at standard quality." },
  "landing.faq.q3":     { fr: "Quelles langues sont supportées ?", en: "Which languages are supported?" },
  "landing.faq.a3":     { fr: "Français et anglais en analyse complète. D'autres langues en bêta.", en: "French and English with full analysis. Other languages in beta." },
  "landing.faq.q4":     { fr: "Puis-je l'utiliser hors ligne ?", en: "Can I use it offline?" },
  "landing.faq.a4":     { fr: "L'application desktop (Windows / macOS) fonctionne hors ligne après l'installation des modèles.", en: "The desktop app (Windows / macOS) works offline once models are installed." },

  "landing.footer.tagline": { fr: "La première plateforme IA dédiée à l'excellence oratoire pour les étudiants et leaders.", en: "The first AI platform built for speaking excellence — for students and leaders." },
  "landing.footer.legal":   { fr: "Légal", en: "Legal" },
  "landing.footer.company": { fr: "Compagnie", en: "Company" },
  "landing.footer.social":  { fr: "Social", en: "Social" },
  "landing.footer.privacy": { fr: "Confidentialité", en: "Privacy" },
  "landing.footer.terms":   { fr: "Conditions", en: "Terms" },
  "landing.footer.contact": { fr: "Contact", en: "Contact" },
  "landing.footer.careers": { fr: "Carrières", en: "Careers" },
  "landing.footer.copy":    { fr: "© 2026 NOOD. Casablanca, Maroc.", en: "© 2026 NOOD. Casablanca, Morocco." },

  // Auth
  "auth.signin.title":  { fr: "Bon retour parmi nous", en: "Welcome back" },
  "auth.signup.title":  { fr: "Créez votre compte", en: "Create your account" },
  "auth.signin.lede":   { fr: "Continuez votre parcours d'orateur.", en: "Continue your speaking journey." },
  "auth.signup.lede":   { fr: "Trois analyses gratuites pour commencer. Pas de carte requise.", en: "Three free analyses to start. No card required." },
  "auth.email":         { fr: "Adresse e-mail", en: "Email" },
  "auth.password":      { fr: "Mot de passe", en: "Password" },
  "auth.fullname":      { fr: "Nom complet", en: "Full name" },
  "auth.signin.cta":    { fr: "Se connecter", en: "Sign in" },
  "auth.signup.cta":    { fr: "Créer mon compte", en: "Create account" },
  "auth.or":            { fr: "ou", en: "or" },
  "auth.google":        { fr: "Continuer avec Google", en: "Continue with Google" },
  "auth.apple":         { fr: "Continuer avec Apple", en: "Continue with Apple" },
  "auth.toggle.signup": { fr: "Pas encore de compte ? Inscrivez-vous", en: "No account yet? Sign up" },
  "auth.toggle.signin": { fr: "Déjà inscrit ? Connectez-vous", en: "Already have an account? Sign in" },
  "auth.forgot":        { fr: "Mot de passe oublié ?", en: "Forgot your password?" },
  "auth.terms":         { fr: "En continuant, vous acceptez nos conditions d'utilisation et notre politique de confidentialité.", en: "By continuing, you agree to our terms and privacy policy." },

  // Workspace / Upload
  "ws.title.a":         { fr: "Par quoi", en: "Where shall we" },
  "ws.title.b":         { fr: "commençons-nous ?", en: "begin?" },
  "ws.upload.drop":     { fr: "Glissez votre enregistrement ici", en: "Drop your recording here" },
  "ws.upload.or":       { fr: "ou", en: "or" },
  "ws.upload.browse":   { fr: "Parcourir", en: "Browse files" },
  "ws.upload.record":   { fr: "Enregistrer maintenant", en: "Record now" },
  "ws.upload.formats":  { fr: "MP4, MOV, WEBM · jusqu'à 500 Mo", en: "MP4, MOV, WEBM · up to 500 MB" },
  "ws.upload.change":   { fr: "Changer de fichier", en: "Change file" },
  "ws.upload.analyze":  { fr: "Analyser ma vidéo", en: "Analyze my video" },
  "ws.context.title":   { fr: "Donnez du contexte à l'IA", en: "Give the AI some context" },
  "ws.context.lede":    { fr: "Plus elle en sait, plus son analyse sera précise.", en: "The more it knows, the sharper the analysis." },
  "ws.context.kind":    { fr: "Type de prise de parole", en: "Type of talk" },
  "ws.context.audience":{ fr: "Audience", en: "Audience" },
  "ws.context.goal":    { fr: "Objectif principal", en: "Primary goal" },
  "ws.context.script":  { fr: "Script ou notes (optionnel)", en: "Script or notes (optional)" },
  "ws.context.script.ph":{ fr: "Collez votre script ou décrivez vos points clés…", en: "Paste your script or describe your key points…" },
  "ws.context.kind.options":      { fr: ["Pitch hackathon", "Soutenance PFE", "Présentation entreprise", "Cours / TD", "Entretien", "Autre"], en: ["Hackathon pitch", "Thesis defense", "Business presentation", "Class / lecture", "Interview", "Other"] },
  "ws.context.audience.options":  { fr: ["Jury technique", "Équipe interne", "Investisseurs", "Étudiants", "Public général"], en: ["Technical jury", "Internal team", "Investors", "Students", "General audience"] },
  "ws.context.goal.options":      { fr: ["Convaincre", "Informer", "Inspirer", "Vendre", "Enseigner"], en: ["Convince", "Inform", "Inspire", "Sell", "Teach"] },
  "ws.priv.encrypt":    { fr: "Vos données sont chiffrées de bout en bout", en: "Your data is end-to-end encrypted" },
  "ws.priv.delete":     { fr: "Supprimable à tout moment", en: "Deletable anytime" },
  "ws.priv.local":      { fr: "Modèles exécutés sur serveur dédié", en: "Models run on dedicated server" },

  // Processing
  "proc.title":         { fr: "Analyse en cours", en: "Analyzing" },
  "proc.lede":          { fr: "Vous pouvez fermer cet onglet — nous vous notifierons.", en: "You can close this tab — we'll notify you." },
  "proc.cancel":        { fr: "Annuler l'analyse", en: "Cancel analysis" },
  "proc.eta":           { fr: "Durée estimée", en: "Estimated time" },
  "proc.elapsed":       { fr: "Temps écoulé", en: "Elapsed" },
  "proc.stage.audio":   { fr: "Extraction audio", en: "Extracting audio" },
  "proc.stage.vad":     { fr: "Détection des pauses", en: "Detecting pauses" },
  "proc.stage.asr":     { fr: "Transcription", en: "Transcription" },
  "proc.stage.prosody": { fr: "Prosodie (hauteur, énergie)", en: "Prosody (pitch, energy)" },
  "proc.stage.vocal":   { fr: "Émotion vocale", en: "Vocal emotion" },
  "proc.stage.body":    { fr: "Langage corporel", en: "Body language" },
  "proc.stage.tone":    { fr: "Adéquation du ton", en: "Tone fit" },
  "proc.stage.score":   { fr: "Calcul du score", en: "Final scoring" },
  "proc.transcript.live":{ fr: "Transcription en direct", en: "Live transcript" },
  "proc.parallel.body": { fr: "Piste corporelle", en: "Body track" },
  "proc.parallel.voice":{ fr: "Piste voix & ton", en: "Voice & tone track" },
  "proc.tip.title":     { fr: "Pendant que vous attendez", en: "While you wait" },
  "proc.tip.body":      { fr: "Les meilleurs orateurs varient leur débit entre 130 et 160 mots/minute. C'est l'écart qui retient l'attention, pas la vitesse.", en: "Top speakers vary pace between 130 and 160 words/minute. It's the variation that holds attention, not the speed." },

  // Report
  "rep.score.total":    { fr: "Score global", en: "Overall score" },
  "rep.grade":          { fr: "Note", en: "Grade" },
  "rep.vs.last":        { fr: "vs. session précédente", en: "vs. last session" },
  "rep.vs.avg":         { fr: "vs. votre moyenne", en: "vs. your average" },
  "rep.summary":        { fr: "Résumé", en: "Summary" },
  "rep.section.body":   { fr: "Langage corporel", en: "Body language" },
  "rep.section.voice":  { fr: "Performance vocale", en: "Vocal performance" },
  "rep.section.tone":   { fr: "Analyse du ton", en: "Tone analysis" },
  "rep.section.content":{ fr: "Contenu & langue", en: "Content & language" },
  "rep.timeline":       { fr: "Chronologie", en: "Timeline" },
  "rep.tips":           { fr: "Conseils de coaching", en: "Coaching tips" },
  "rep.tips.lede":      { fr: "Classés par impact. Cliquez pour voir le moment.", en: "Ranked by impact. Click to jump to the moment." },
  "rep.tips.done":      { fr: "Marquer comme travaillé", en: "Mark as worked on" },
  "rep.video":          { fr: "Lecture", en: "Playback" },
  "rep.transcript":     { fr: "Transcription", en: "Transcript" },
  "rep.export":         { fr: "Exporter", en: "Export" },
  "rep.share":          { fr: "Partager", en: "Share" },
  "rep.again":          { fr: "Nouvelle analyse", en: "New analysis" },
  "rep.metric.wpm":     { fr: "Débit", en: "Pace" },
  "rep.metric.fillers": { fr: "Mots parasites", en: "Fillers" },
  "rep.metric.pitch":   { fr: "Variation de hauteur", en: "Pitch variation" },
  "rep.metric.energy":  { fr: "Variation d'énergie", en: "Energy variation" },
  "rep.metric.pause":   { fr: "Ratio de pauses", en: "Pause ratio" },
  "rep.metric.emotion": { fr: "Émotion vocale", en: "Vocal emotion" },
  "rep.dominant":       { fr: "Émotion dominante", en: "Dominant emotion" },
  "rep.frames":         { fr: "images analysées", en: "frames analyzed" },
  "rep.tone.topic":     { fr: "Sujet détecté", en: "Detected topic" },
  "rep.tone.context":   { fr: "Contexte", en: "Context" },
  "rep.tone.fit":       { fr: "Adéquation du ton", en: "Tone fit" },
  "rep.tone.appropriate":{ fr: "Approprié", en: "Appropriate" },
  "rep.mismatches":     { fr: "Décalages détectés", en: "Mismatches detected" },
  "rep.severity.high":  { fr: "Élevé", en: "High" },
  "rep.severity.med":   { fr: "Moyen", en: "Medium" },
  "rep.severity.low":   { fr: "Faible", en: "Low" },
  "rep.observed":       { fr: "Observé", en: "Observed" },
  "rep.expected":       { fr: "Attendu", en: "Expected" },
  "rep.next":           { fr: "À pratiquer cette semaine", en: "Practice this week" },

  // History
  "hist.title":         { fr: "Vos sessions", en: "Your sessions" },
  "hist.lede":          { fr: "Suivez vos progrès dans le temps.", en: "Track your progress over time." },
  "hist.new":           { fr: "Nouvelle analyse", en: "New analysis" },
  "hist.trend":         { fr: "Score moyen — 30 derniers jours", en: "Avg. score — last 30 days" },
  "hist.compare":       { fr: "Comparer", en: "Compare" },

  // Common
  "common.minutes":     { fr: "min", en: "min" },
  "common.seconds":     { fr: "s", en: "s" },
  "common.back":        { fr: "Retour", en: "Back" },
  "common.continue":    { fr: "Continuer", en: "Continue" },
  "common.cancel":      { fr: "Annuler", en: "Cancel" },
  "common.close":       { fr: "Fermer", en: "Close" },
  "common.skip":        { fr: "Ignorer", en: "Skip" },
};

const I18nContext = React.createContext({ lang: 'fr', t: (k) => k, setLang: () => {} });

function I18nProvider({ children, defaultLang = 'fr' }) {
  const [lang, setLang] = React.useState(defaultLang);
  const t = React.useCallback((key) => {
    const entry = STRINGS[key];
    if (!entry) return key;
    return entry[lang] ?? entry.fr ?? key;
  }, [lang]);
  return (
    <I18nContext.Provider value={{ lang, setLang, t }}>
      {children}
    </I18nContext.Provider>
  );
}

const useT = () => React.useContext(I18nContext);

Object.assign(window, { I18nProvider, useT, STRINGS });
