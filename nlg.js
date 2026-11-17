// ------------------------------------------------------------
// TEMPLATE-BASED NLG (Natural Language Generation)
// ------------------------------------------------------------
// Slot-filled templates that produce varied, grammatical feedback.
// Deterministic (seeded per rep), offline, no API.
//
// Each template supports slots in {braces}. Slot values are supplied
// by the caller. Missaing slots render as `{slotName}` so bugs are
// visible rather than silent.
//
// Design notes:
// - Randomness is seeded per-rep, not per-frame, to avoid text flickering
//   60x/sec while the user holds a position.
// - Per-exercise vocabulary (deeperWord, standWord, targetWord) lets
//   squat say "deeper"/"stand" while knee-flexion says "flex"/"extend".
// - Every template is a single sentence ending in a period.

const TEMPLATES = {
    // ---- Squat-style descent cues ----
    deeper: [
        "Try going a little deeper — aim for about {target}° at the knee.",
        "You're {delta}° above target depth. {deeperWord} a bit more.",
        "Almost there — {deeperWord} {delta}° more next rep."
    ],

    // ---- Standing / extension cues ----
    standTall: [
        "Stand tall at the top — extend to about {target}°.",
        "You're {delta}° short of full extension. {standWord} up.",
        "Push up to {target}° — you're {delta}° away."
    ],

    // ---- Good rep, mid-rep ----
    goodDepth: [
        "Good depth — {min}° at the bottom. Now {standWord}.",
        "Nice, you hit {min}°. {standWord} back up."
    ],

    // ---- Tempo ----
    slowDown: [
        "That rep took {dur}s. Slow down for a controlled {targetDur}s cycle.",
        "Too fast — aim for {targetDur}s per rep.",
        "You're rushing. Target {targetDur}s per rep."
    ],

    // ---- Torso lean ----
    torsoLean: [
        "Keep your chest up — torso leaned {deg}°, limit is {limit}°.",
        "You're leaning forward {deg}°. Brace your core.",
        "Torso tilt hit {deg}° — stay under {limit}°."
    ],

    // ---- Hip drop (squat) ----
    hipsHigh: [
        "Hips only dropped {drop}% — aim for at least {targetDrop}%.",
        "Sit back further; your hips barely moved ({drop}%).",
        "Deeper hip drop needed — currently {drop}%, target {targetDrop}%."
    ],

    // ---- Seated posture (knee flexion) ----
    notSeated: [
        "Sit upright — thighs should be roughly level with your hips.",
        "Adjust your seat — knees and hips should be at the same height.",
        "You're not seated squarely. Settle back into the chair."
    ],

    // ---- Knees over ankles (sit-to-stand) ----
    kneesNotOverAnkles: [
        "Keep your knees over your ankles — you're drifting forward.",
        "Knees and ankles should line up. Shift back slightly.",
        "Your knees are ahead of your ankles. Sit back a touch."
    ],

    // ---- Excessive torso lean (STS) ----
    excessiveLean: [
        "Too much forward lean ({deg}°). Push through your legs more.",
        "You're hinging at the hips ({deg}°). Stand more upright.",
        "Lean is {deg}° — try to stay under {limit}°."
    ],

    // ---- Calibration ----
    calibrating: [
        "Hold your starting position for a moment so I can calibrate.",
        "Calibrating — stay still in the start pose.",
        "One moment — establishing your baseline."
    ],

    // ---- No pose / angle ----
    adjustPosition: [
        "Adjust your position so your knees and hips are fully in frame.",
        "I can't see your knees clearly. Step back or turn slightly.",
        "Make sure your full lower body is visible to the camera."
    ],

    // ---- Rep-complete summary (post-rep) ----
    repSummaryGood: [
        "Rep {rep}: strong — {min}° depth, {dur}s tempo.",
        "Rep {rep}: solid form ({min}° bottom, {dur}s cycle).",
        "Rep {rep}: good rep. Keep it up."
    ],

    repSummaryFair: [
        "Rep {rep}: fair — {min}° bottom. Aim for {target}° next time.",
        "Rep {rep}: okay. {standWord} a bit more and control the tempo.",
        "Rep {rep}: acceptable, but room to improve."
    ],

    repSummaryPoor: [
        "Rep {rep}: needs work. {cueText}",
        "Rep {rep}: not counted as quality. {cueText}",
        "Rep {rep}: let's reset and try again."
    ]
};

// ------------------------------------------------------------
// PER-EXERCISE VOCABULARY
// ------------------------------------------------------------
// The same NLG module produces correct verbs per exercise.
//
// IMPORTANT: verbs map to rep-state semantics, not physical motion.
//   - "DOWN" state = avg angle below downAngle
//   - "UP"   state = avg angle above upAngle
//
// So for kneeFlexion:
//   - DOWN = knee flexed → deeperWord = "Flex"
//   - UP   = leg extended → standWord  = "Extend"

const VOCAB = {
    squat: {
        deeperWord: "Sink",
        standWord: "Stand",
        targetWord: "depth"
    },
    sitToStand: {
        deeperWord: "Sit back",
        standWord: "Stand",
        targetWord: "seated angle"
    },
    kneeFlexion: {
        deeperWord: "Flex",
        standWord: "Extend",
        targetWord: "flexion"
    }
};

export function vocabFor(exerciseId) {
    return VOCAB[exerciseId] || VOCAB.squat;
}

// ------------------------------------------------------------
// RENDERING
// ------------------------------------------------------------

function fill(template, slots) {
    return template.replace(/\{(\w+)\}/g, (_, k) =>
        slots[k] !== undefined && slots[k] !== null ? slots[k] : `{${k}}`
    );
}

// Deterministic seeded selection so text doesn't flicker frame-to-frame.
function seededPick(list, seed) {
    if (!list || list.length === 0) return null;
    const idx = Math.abs(Math.floor(seed || 0)) % list.length;
    return list[idx];
}

export function generateFeedback(kind, slots = {}, seed = 0) {
    const list = TEMPLATES[kind];
    if (!list) return "";
    return fill(seededPick(list, seed), slots);
}

export function generateRepSummary(grade, slots = {}, seed = 0) {
    const kind =
        grade === "good" ? "repSummaryGood" :
            grade === "fair" ? "repSummaryFair" :
                "repSummaryPoor";
    return generateFeedback(kind, slots, seed);
}

// For tests / thesis appendix.
export function templatesFor(kind) {
    return TEMPLATES[kind] ? [...TEMPLATES[kind]] : [];
}

export function allTemplateKinds() {
    return Object.keys(TEMPLATES);
}