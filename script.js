import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs";

import { generateFeedback, generateRepSummary, vocabFor } from "./nlg.js";

// ------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------

const MP_VERSION = "0.10.12";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// CALIBRATION: smoothing factor. Cite or run a sensitivity analysis.
const SMOOTHING = 0.7;
const MIN_VISIBILITY = 0.5;

const DOWN_HOLD_MS = 100;
const UP_HOLD_MS = 100;

const POSE_LOST_RESET_FRAMES = 15;
const MAX_REP_DURATION_MS = 8000;

const FLASH_DURATION_MS = 2500;

// Frame logging is downsampled to bound sessionLog memory.
const FRAME_LOG_INTERVAL_MS = 100; // ~10 Hz

// ------------------------------------------------------------
// ANTI-CHEAT / POSTURE VALIDATION
// ------------------------------------------------------------

// CALIBRATION: these are default heuristics. Justify each in your methods.
const MIN_HIP_DROP_RATIO = 0.12;
const MAX_TORSO_TILT_DEG = 35;
const BASELINE_HOLD_MS = 800;
const BASELINE_LOST_FRAMES = 30;

// ------------------------------------------------------------
// TEMPO SCORING (configurable per exercise)
// ------------------------------------------------------------
// Default bands assume controlled, moderate-tempo reps.
// TODO: cite a rehab protocol and replace per exercise.
const DEFAULT_TEMPO_BANDS = [
  [1000, 30],
  [1500, 60],
  [4000, 100],
  [6000, 60],
  [Infinity, 30]
];

// ------------------------------------------------------------
// SHARED GEOMETRY HELPERS
// ------------------------------------------------------------

function calculateAngle(a, b, c) {
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;

  const dot = abx * cbx + aby * cby;
  const magAB = Math.hypot(abx, aby);
  const magCB = Math.hypot(cbx, cby);

  if (magAB < 1e-6 || magCB < 1e-6) return null;

  let cosine = dot / (magAB * magCB);
  cosine = Math.max(-1, Math.min(1, cosine));

  return Math.acos(cosine) * (180 / Math.PI);
}

function calculateTorsoTilt(shoulder, hip) {
  if (!shoulder || !hip) return null;
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y;
  if (Math.hypot(dx, dy) < 1e-6) return null;
  return Math.atan2(Math.abs(dx), Math.abs(dy)) * (180 / Math.PI);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// ------------------------------------------------------------
// SHARED BASELINE TRACKER
// ------------------------------------------------------------
// Baselines are stored in CANVAS-PIXEL coordinates (same space as
// `points` from smoothPoint), NOT MediaPipe's normalized landmarks.

function createBaselineTracker({
  isReferencePose,
  measureReference,
  onLost
}) {
  let baseline = null;
  let since = null;
  let lostFrames = 0;

  return {
    update(points, avgAngle, now) {
      const isRef = isReferencePose(points, avgAngle);

      if (!isRef) {
        since = null;
        lostFrames++;
        if (baseline !== null && lostFrames > BASELINE_LOST_FRAMES * 3) {
          baseline = null;
          if (onLost) onLost();
        }
        return;
      }

      lostFrames = 0;
      if (since === null) since = now;

      if (now - since >= BASELINE_HOLD_MS) {
        const measured = measureReference(points);
        if (measured) {
          if (baseline === null) {
            baseline = measured;
          } else {
            const next = {};
            for (const k of Object.keys(measured)) {
              const prev = baseline[k];
              next[k] = prev === undefined
                ? measured[k]
                : prev * 0.9 + measured[k] * 0.1;
            }
            baseline = next;
          }
        }
      }
    },

    get() {
      return baseline;
    },

    reset() {
      baseline = null;
      since = null;
      lostFrames = 0;
    }
  };
}

// ------------------------------------------------------------
// EXERCISE DEFINITIONS
// ------------------------------------------------------------

const EXERCISES = {
  squat: (() => {
    const baseline = createBaselineTracker({
      isReferencePose: (points, avg) =>
        avg !== null &&
        avg > 160 &&
        points[23] && points[24] && points[27] && points[28],

      measureReference(points) {
        const lHip = points[23], rHip = points[24];
        const lAnkle = points[27], rAnkle = points[28];
        const hip = midpoint(lHip, rHip);
        const ankle = midpoint(lAnkle, rAnkle);
        return {
          hipCanvasY: hip.y,
          legLengthPx: Math.hypot(hip.x - ankle.x, hip.y - ankle.y)
        };
      }
    });

    return {
      id: "squat",
      name: "Squat",
      labelPlural: "Squats",
      subtitle: "Position your full lower body in view, then start the camera.",
      downAngle: 120,
      upAngle: 165,
      tempoBands: DEFAULT_TEMPO_BANDS,

      measure(points) {
        const lHip = points[23], rHip = points[24];
        const lKnee = points[25], rKnee = points[26];
        const lAnkle = points[27], rAnkle = points[28];

        if (!lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) {
          return null;
        }

        const lAngle = calculateAngle(lHip, lKnee, lAnkle);
        const rAngle = calculateAngle(rHip, rKnee, rAnkle);
        if (lAngle === null || rAngle === null) return null;

        return { left: lAngle, right: rAngle, avg: (lAngle + rAngle) / 2 };
      },

      updateBaseline(points, avg, now) {
        baseline.update(points, avg, now);
      },

      validate(points, avg) {
        if (avg === null) return { valid: false, reason: "no-angle" };

        const lHip = points[23], rHip = points[24];
        const lAnkle = points[27], rAnkle = points[28];
        const lShoulder = points[11], rShoulder = points[12];

        if (!lHip || !rHip || !lAnkle || !rAnkle) {
          return { valid: false, reason: "missing-points" };
        }

        if (!(lAnkle.y > lHip.y + 20 && rAnkle.y > rHip.y + 20)) {
          return { valid: false, reason: "ankle-above-hip" };
        }

        const base = baseline.get();
        let hipDropPct = null;

        if (base && base.hipCanvasY !== null && base.legLengthPx !== null) {
          const hipY = (lHip.y + rHip.y) / 2;
          const hipDrop = hipY - base.hipCanvasY;
          const requiredDrop = base.legLengthPx * MIN_HIP_DROP_RATIO;

          hipDropPct = base.legLengthPx > 0
            ? Math.round((hipDrop / base.legLengthPx) * 100)
            : 0;

          if (avg < this.downAngle && hipDrop < requiredDrop) {
            return { valid: false, reason: "hips-too-high", hipDropPct };
          }
        }

        if (lShoulder && rShoulder) {
          const lTilt = calculateTorsoTilt(lShoulder, lHip);
          const rTilt = calculateTorsoTilt(rShoulder, rHip);
          if (lTilt !== null && rTilt !== null) {
            const avgTilt = (lTilt + rTilt) / 2;
            if (avgTilt > MAX_TORSO_TILT_DEG) {
              return { valid: false, reason: "torso-lean" };
            }
          }
        }

        return { valid: true, reason: "ok", hipDropPct };
      },

      reset() {
        baseline.reset();
      }
    };
  })(),

  sitToStand: (() => {
    // CALIBRATION: verify against your cited STS protocol.
    const SEATED_ANGLE_MAX = 110;
    const STANDING_ANGLE_MIN = 160;

    const baseline = createBaselineTracker({
      isReferencePose: (points, avg) =>
        avg !== null &&
        avg < SEATED_ANGLE_MAX &&
        points[23] && points[24] && points[27] && points[28],

      measureReference(points) {
        const lHip = points[23], rHip = points[24];
        return { seatedHipCanvasY: midpoint(lHip, rHip).y };
      }
    });

    return {
      id: "sitToStand",
      name: "Sit-to-Stand",
      labelPlural: "Sit-to-Stands",
      subtitle: "Sit on a chair, side-on to the camera. Stand up fully, then sit back down.",
      downAngle: SEATED_ANGLE_MAX,
      upAngle: STANDING_ANGLE_MIN,
      tempoBands: DEFAULT_TEMPO_BANDS,

      measure(points) {
        return EXERCISES.squat.measure(points);
      },

      updateBaseline(points, avg, now) {
        baseline.update(points, avg, now);
      },

      validate(points, avg) {
        if (avg === null) return { valid: false, reason: "no-angle" };

        const lHip = points[23], rHip = points[24];
        const lKnee = points[25], rKnee = points[26];
        const lAnkle = points[27], rAnkle = points[28];

        if (!lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) {
          return { valid: false, reason: "missing-points" };
        }

        const kneeOverAnkle =
          Math.abs(lKnee.x - lAnkle.x) < Math.abs(lKnee.y - lAnkle.y) * 1.5 &&
          Math.abs(rKnee.x - rAnkle.x) < Math.abs(rKnee.y - rAnkle.y) * 1.5;
        if (!kneeOverAnkle) {
          return { valid: false, reason: "knees-not-over-ankles" };
        }

        if (avg < this.downAngle && baseline.get() === null) {
          return { valid: false, reason: "calibrating" };
        }

        const lShoulder = points[11], rShoulder = points[12];
        if (lShoulder && rShoulder) {
          const lTilt = calculateTorsoTilt(lShoulder, lHip);
          const rTilt = calculateTorsoTilt(rShoulder, rHip);
          if (lTilt !== null && rTilt !== null) {
            // CALIBRATION: 55° is permissive; STS naturally leans ~30-45°.
            if ((lTilt + rTilt) / 2 > 55) {
              return { valid: false, reason: "excessive-torso-lean" };
            }
          }
        }

        return { valid: true, reason: "ok" };
      },

      reset() {
        baseline.reset();
      }
    };
  })(),

  kneeFlexion: (() => {
    // CALIBRATION: verify against your cited knee-flexion protocol.
    const SEATED_ANGLE_MAX = 100;
    const EXTENDED_ANGLE_MIN = 160;

    const baseline = createBaselineTracker({
      isReferencePose: (points, avg) =>
        avg !== null &&
        avg < SEATED_ANGLE_MAX &&
        points[23] && points[24] && points[25] && points[26],

      measureReference(points) {
        const lHip = points[23], rHip = points[24];
        const lKnee = points[25], rKnee = points[26];
        const lThighTilt = calculateTorsoTilt(lHip, lKnee);
        const rThighTilt = calculateTorsoTilt(rHip, rKnee);
        return {
          thighTiltDeg: (lThighTilt + rThighTilt) / 2
        };
      }
    });

    return {
      id: "kneeFlexion",
      name: "Knee Flexion",
      labelPlural: "Knee Flexions",
      subtitle: "Sit on a chair facing the camera side-on. Extend your leg fully, then flex back.",
      downAngle: SEATED_ANGLE_MAX,
      upAngle: EXTENDED_ANGLE_MIN,
      tempoBands: DEFAULT_TEMPO_BANDS,

      measure(points) {
        return EXERCISES.squat.measure(points);
      },

      updateBaseline(points, avg, now) {
        baseline.update(points, avg, now);
      },

      validate(points, avg) {
        if (avg === null) return { valid: false, reason: "no-angle" };

        const lHip = points[23], rHip = points[24];
        const lKnee = points[25], rKnee = points[26];
        const lAnkle = points[27], rAnkle = points[28];

        if (!lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) {
          return { valid: false, reason: "missing-points" };
        }

        const avgHipY = (lHip.y + rHip.y) / 2;
        const avgKneeY = (lKnee.y + rKnee.y) / 2;
        // CALIBRATION: 40 canvas-px tolerance for "seated".
        if (Math.abs(avgHipY - avgKneeY) > 40) {
          return { valid: false, reason: "not-seated" };
        }

        return { valid: true, reason: "ok" };
      },

      reset() {
        baseline.reset();
      }
    };
  })()
};

let currentExercise = EXERCISES.squat;

// ------------------------------------------------------------
// DOM ELEMENTS
// ------------------------------------------------------------

const sourceVideo = document.getElementById("sourceVideo");
const stageCanvas = document.getElementById("stageCanvas");
const stageMessage = document.getElementById("stageMessage");

const openBtn = document.getElementById("openBtn");
const closeBtn = document.getElementById("closeBtn");
const exportBtn = document.getElementById("exportBtn");

const statusEl = document.getElementById("status");
const mpStatusEl = document.getElementById("mpStatus");

const repLabelEl = document.getElementById("repLabel");
const repCountEl = document.getElementById("repCount");
const repStateEl = document.getElementById("repState");
const angleValueEl = document.getElementById("angleValue");
const lrAnglesEl = document.getElementById("lrAngles");

const feedbackEl = document.getElementById("feedback");
const flashEl = document.getElementById("flash");
const flashTextEl = document.getElementById("flashText");

const targetRepsEl = document.getElementById("targetReps");
const progressBarEl = document.getElementById("progressBar");
const progressTextEl = document.getElementById("progressText");
const progressStatusEl = document.getElementById("progressStatus");

const lastCommentEl = document.getElementById("lastComment");
const lastCommentTextEl = document.getElementById("lastCommentText");

const sessionCompleteEl = document.getElementById("sessionComplete");

const evalStatusEl = document.getElementById("evalStatus");
const evalDepthEl = document.getElementById("evalDepth");
const evalTempoEl = document.getElementById("evalTempo");
const evalRangeEl = document.getElementById("evalRange");
const evalOverallEl = document.getElementById("evalOverall");
const evalMinAngleEl = document.getElementById("evalMinAngle");
const evalMaxAngleEl = document.getElementById("evalMaxAngle");
const evalDurationEl = document.getElementById("evalDuration");
const evalReasonEl = document.getElementById("evalReason");

const exerciseSelectEl = document.getElementById("exerciseSelect");
const appSubtitleEl = document.getElementById("appSubtitle");

const ctx = stageCanvas.getContext("2d");

// ------------------------------------------------------------
// STATE
// ------------------------------------------------------------

let stream = null;
let rafId = null;
let poseLandmarker = null;

let modelReady = false;
let cameraStarting = false;
let cameraSession = 0;

let lastVideoTime = -1;
let latestLandmarks = null;

let previousPoints = {};
let missingFramesByLandmark = {};

let repCount = 0;
let repState = "UP";

let leftKneeAngle = null;
let rightKneeAngle = null;

let downSince = null;
let upSince = null;

let poseLostFrames = 0;
let repStartTime = null;
let repStartReason = null; // "descent-onset" | "down-transition"

let repMinAngle = null;
let repMaxAngle = null;

let sessionLog = [];
let sessionStartTime = null;
let lastFrameLogAt = 0;

let flashUntil = 0;

let lastEvaluation = null;
let lastEvaluationAt = 0;

// NLG input caches — populated each frame from validation.
let lastValidation = null;
let lastTorsoDeg = null;
let lastHipDropPct = null;

// Target + progress
let targetReps = 10;
let sessionCompleteShown = false;

// ------------------------------------------------------------
// LANDMARK MAPS
// ------------------------------------------------------------

const LOWER_BODY_INDICES = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const DRAWN_JOINT_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

const LOWER_BODY_CONNECTIONS = [
  [23, 25], [25, 27],
  [24, 26], [26, 28],
  [23, 24], [25, 26],
  [27, 28], [27, 29],
  [28, 30], [29, 31],
  [30, 32]
];

// ------------------------------------------------------------
// UI HELPERS
// ------------------------------------------------------------

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", isError);
}

function setMpStatus(message, isError = false) {
  mpStatusEl.textContent = message;
  mpStatusEl.classList.toggle("is-error", isError);
}

function setCameraUi(isOn) {
  openBtn.disabled = isOn || cameraStarting;
  closeBtn.disabled = !isOn && !cameraStarting;
  stageMessage.classList.toggle("is-hidden", isOn);
}

// ---- NLG wiring ----

function buildFeedbackSlots(avg) {
  const v = vocabFor(currentExercise.id);
  const durMs = repStartTime !== null ? performance.now() - repStartTime : 0;
  const middleBand = currentExercise.tempoBands?.[2]?.[0] ?? 4000;

  return {
    ...v,
    target: Math.round(currentExercise.downAngle),
    targetDur: (middleBand / 1000).toFixed(1),
    delta: Math.abs(Math.round((avg ?? 0) - currentExercise.downAngle)),
    min: Math.round(repMinAngle ?? avg ?? 0),
    dur: (durMs / 1000).toFixed(1),
    deg: Math.round(lastTorsoDeg ?? 0),
    limit: MAX_TORSO_TILT_DEG,
    drop: Math.round(lastHipDropPct ?? 0),
    targetDrop: Math.round(MIN_HIP_DROP_RATIO * 100),
    rep: repCount
  };
}

function buildLiveFeedback(avg) {
  const slots = buildFeedbackSlots(avg);
  let kind = null;

  if (lastValidation && !lastValidation.valid) {
    switch (lastValidation.reason) {
      case "torso-lean": kind = "torsoLean"; break;
      case "excessive-torso-lean": kind = "excessiveLean"; break;
      case "hips-too-high": kind = "hipsHigh"; break;
      case "not-seated": kind = "notSeated"; break;
      case "knees-not-over-ankles": kind = "kneesNotOverAnkles"; break;
      case "calibrating": kind = "calibrating"; break;
      case "missing-points":
      case "no-angle": kind = "adjustPosition"; break;
      default: kind = null;
    }
  } else if (avg !== null) {
    if (repState === "UP" && avg < currentExercise.upAngle - 5) {
      kind = "standTall";
    } else if (repState === "DOWN" && avg > currentExercise.downAngle - 5) {
      kind = "deeper";
    } else if (repState === "DOWN") {
      kind = "goodDepth";
    }
  }

  if (!kind) {
    return generateFeedback("adjustPosition", slots, repCount);
  }
  return generateFeedback(kind, slots, repCount);
}

function updatePanel() {
  repLabelEl.textContent = currentExercise.labelPlural || "Reps";
  repCountEl.textContent = String(repCount);
  repStateEl.textContent = repState;

  angleValueEl.textContent =
    leftKneeAngle !== null && rightKneeAngle !== null
      ? `${Math.round((leftKneeAngle + rightKneeAngle) / 2)}°`
      : "--";

  lrAnglesEl.textContent =
    `${leftKneeAngle === null ? "--" : Math.round(leftKneeAngle) + "°"} / ` +
    `${rightKneeAngle === null ? "--" : Math.round(rightKneeAngle) + "°"}`;

  const avg =
    leftKneeAngle !== null && rightKneeAngle !== null
      ? (leftKneeAngle + rightKneeAngle) / 2
      : null;

  feedbackEl.textContent = buildLiveFeedback(avg);
}

function updateProgress() {
  const pct = targetReps > 0
    ? Math.min(100, (repCount / targetReps) * 100)
    : 0;

  progressBarEl.style.width = `${pct}%`;
  progressTextEl.textContent = `${repCount} / ${targetReps}`;

  const done = repCount >= targetReps;

  progressBarEl.classList.toggle("is-complete", done);
  progressStatusEl.classList.toggle("is-complete", done);
  progressStatusEl.textContent = done ? "Complete" : "In progress";

  if (done && !sessionCompleteShown) {
    sessionCompleteEl.classList.remove("is-hidden");
    sessionCompleteShown = true;
  } else if (!done) {
    sessionCompleteEl.classList.add("is-hidden");
    sessionCompleteShown = false;
  }
}

function showFlash(text, grade) {
  flashTextEl.textContent = text;
  flashEl.classList.remove("is-hidden", "flash--good", "flash--fair", "flash--poor");
  if (grade) flashEl.classList.add(`flash--${grade}`);
  flashUntil = performance.now() + FLASH_DURATION_MS;
}

function updateFlashVisibility(now) {
  if (flashUntil === 0) return;
  if (now > flashUntil) {
    flashEl.classList.add("is-hidden");
    flashUntil = 0;
  }
}

function escapeCsv(str) {
  return String(str || "").replace(/"/g, '""');
}

function sessionTimeSec(now) {
  if (sessionStartTime === null) return 0;
  return (now - sessionStartTime) / 1000;
}

// ------------------------------------------------------------
// STATE RESET HELPERS
// ------------------------------------------------------------

function resetRepState() {
  downSince = null;
  upSince = null;
  repStartTime = null;
  repStartReason = null;
  repMinAngle = null;
  repMaxAngle = null;
}

function resetTrackingState() {
  previousPoints = {};
  missingFramesByLandmark = {};

  latestLandmarks = null;
  lastVideoTime = -1;

  leftKneeAngle = null;
  rightKneeAngle = null;

  repState = "UP";
  resetRepState();

  poseLostFrames = 0;

  lastEvaluation = null;
  lastEvaluationAt = 0;

  lastValidation = null;
  lastTorsoDeg = null;
  lastHipDropPct = null;

  lastCommentEl.classList.add("is-hidden");
  lastCommentTextEl.textContent = "";

  sessionCompleteEl.classList.add("is-hidden");
  sessionCompleteShown = false;

  lastFrameLogAt = 0;

  currentExercise.reset?.();

  updateEvaluationPanel(null);
}

// ------------------------------------------------------------
// MEDIAPIPE INIT
// ------------------------------------------------------------

async function initPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.7
  });

  return poseLandmarker;
}

// ------------------------------------------------------------
// GEOMETRY / SMOOTHING
// ------------------------------------------------------------

function isLandmarkVisible(landmark) {
  return Boolean(
    landmark &&
    Number.isFinite(landmark.x) &&
    Number.isFinite(landmark.y) &&
    (landmark.visibility === undefined ||
      landmark.visibility >= MIN_VISIBILITY)
  );
}

function smoothPoint(index, rawX, rawY) {
  if (missingFramesByLandmark[index] > POSE_LOST_RESET_FRAMES) {
    previousPoints[index] = { x: rawX, y: rawY };
  }
  missingFramesByLandmark[index] = 0;

  if (!previousPoints[index]) {
    previousPoints[index] = { x: rawX, y: rawY };
    return previousPoints[index];
  }

  previousPoints[index].x =
    previousPoints[index].x * SMOOTHING + rawX * (1 - SMOOTHING);
  previousPoints[index].y =
    previousPoints[index].y * SMOOTHING + rawY * (1 - SMOOTHING);

  return previousPoints[index];
}

function markLandmarkMissing(index) {
  missingFramesByLandmark[index] =
    (missingFramesByLandmark[index] || 0) + 1;
}

// ------------------------------------------------------------
// POSE-LOST HANDLING
// ------------------------------------------------------------

function handlePoseLostFrame() {
  poseLostFrames++;
  if (poseLostFrames >= POSE_LOST_RESET_FRAMES) {
    repState = "UP";
    resetRepState();
  }
}

function handlePoseFoundFrame() {
  poseLostFrames = 0;
}

// ------------------------------------------------------------
// EVALUATION
// ------------------------------------------------------------

function scoreDepth(minAngle) {
  if (minAngle === null) return 0;
  if (minAngle <= 90) return 100;
  if (minAngle >= 120) return 0;
  return 100 - ((minAngle - 90) / 30) * 100;
}

function scoreTempo(durationMs, bands) {
  for (const [maxMs, score] of bands) {
    if (durationMs < maxMs) return score;
  }
  return 0;
}

function scoreRange(minAngle, maxAngle) {
  if (minAngle === null || maxAngle === null) return 0;
  const range = maxAngle - minAngle;
  if (range >= 70) return 100;
  if (range <= 20) return 0;
  return ((range - 20) / 50) * 100;
}

function gradeFor(overall) {
  if (overall >= 80) return "good";
  if (overall >= 60) return "fair";
  return "poor";
}

// Short rule-based cue used ONLY as a slot value inside repSummaryPoor.
// The full sentence is still generated by NLG; this is just the "what
// went wrong" fragment.
function cueFor(minAngle, maxAngle, durationMs) {
  if (minAngle === null) return "Rep incomplete";
  const range = (minAngle !== null && maxAngle !== null)
    ? maxAngle - minAngle
    : null;
  if (minAngle > 105) return "go deeper next rep";
  if (range !== null && range < 40) return "increase range of motion";
  if (durationMs < 1200) return "slow down — controlled movement";
  return "keep the same form";
}

function round1(n) {
  return n === null ? null : Math.round(n * 10) / 10;
}

function evaluateRep({
  minAngle,
  maxAngle,
  durationMs,
  partial = false,
  bands = currentExercise.tempoBands
}) {
  const depthScore = minAngle !== null ? scoreDepth(minAngle) : null;
  const tempoScore = durationMs > 0 ? scoreTempo(durationMs, bands) : null;
  const rangeScore =
    (minAngle !== null && maxAngle !== null)
      ? scoreRange(minAngle, maxAngle)
      : null;

  const complete =
    depthScore !== null && tempoScore !== null && rangeScore !== null;

  const overall = complete
    ? Math.round((depthScore + tempoScore + rangeScore) / 3)
    : null;

  return {
    partial,
    minAngle: round1(minAngle),
    maxAngle: round1(maxAngle),
    durationMs: Math.round(durationMs || 0),
    depthScore: depthScore !== null ? Math.round(depthScore) : null,
    tempoScore: tempoScore !== null ? Math.round(tempoScore) : null,
    rangeScore: rangeScore !== null ? Math.round(rangeScore) : null,
    overall,
    grade: overall !== null ? gradeFor(overall) : null,
    cue: cueFor(minAngle, maxAngle, durationMs)
  };
}

function snapshotEvaluation(avg, validation, now) {
  const minA = repMinAngle ?? avg;
  const maxA = repMaxAngle ?? avg;
  const dur = repStartTime !== null ? now - repStartTime : 0;

  const result = evaluateRep({
    minAngle: minA,
    maxAngle: maxA,
    durationMs: dur,
    partial: true
  });

  let status = "idle";
  let reason = "";

  if (!validation || !validation.valid) {
    if (validation) {
      const map = {
        "torso-lean": "Torso lean detected",
        "hips-too-high": "Hips didn't drop enough",
        "excessive-torso-lean": "Excessive torso lean",
        "not-seated": "Not in seated posture",
        "knees-not-over-ankles": "Knees drifting forward"
      };
      if (validation.reason === "calibrating") {
        status = "calibrating";
      } else if (map[validation.reason]) {
        status = "rejected";
        reason = map[validation.reason];
      } else if (
        validation.reason !== "no-angle" &&
        validation.reason !== "missing-points"
      ) {
        status = "rejected";
        reason = validation.reason;
      }
    }
  } else if (repState === "DOWN") {
    status = "counting";
  }

  return {
    status,
    reason,
    depth: result.depthScore,
    tempo: result.tempoScore,
    range: result.rangeScore,
    overall: result.overall,
    minAngle: result.minAngle,
    maxAngle: result.maxAngle,
    durationMs: result.durationMs,
    valid: validation ? validation.valid : false
  };
}

function updateEvaluationPanel(snapshot) {
  if (!snapshot) {
    evalStatusEl.textContent = "Idle";
    evalStatusEl.dataset.status = "idle";
    evalDepthEl.textContent = "--";
    evalTempoEl.textContent = "--";
    evalRangeEl.textContent = "--";
    evalOverallEl.textContent = "--";
    evalMinAngleEl.textContent = "--";
    evalMaxAngleEl.textContent = "--";
    evalDurationEl.textContent = "--";
    evalReasonEl.textContent = "";
    return;
  }

  const statusLabels = {
    idle: "Idle",
    counting: "Counting…",
    rejected: "Rejected",
    calibrating: "Calibrating…"
  };

  evalStatusEl.textContent = statusLabels[snapshot.status] || "Idle";
  evalStatusEl.dataset.status = snapshot.status;

  evalDepthEl.textContent = snapshot.depth !== null ? snapshot.depth : "--";
  evalTempoEl.textContent = snapshot.tempo !== null ? snapshot.tempo : "--";
  evalRangeEl.textContent = snapshot.range !== null ? snapshot.range : "--";
  evalOverallEl.textContent = snapshot.overall !== null ? snapshot.overall : "--";

  evalMinAngleEl.textContent =
    snapshot.minAngle !== null ? `${Math.round(snapshot.minAngle)}°` : "--";
  evalMaxAngleEl.textContent =
    snapshot.maxAngle !== null ? `${Math.round(snapshot.maxAngle)}°` : "--";
  evalDurationEl.textContent =
    snapshot.durationMs > 0
      ? `${(snapshot.durationMs / 1000).toFixed(1)}s`
      : "--";

  evalReasonEl.textContent = snapshot.reason || "";
}

function finalizeRepEvaluation(now, { minAngle, maxAngle, durationMs }) {
  const result = evaluateRep({ minAngle, maxAngle, durationMs, partial: false });

  const slots = {
    ...buildFeedbackSlots(minAngle),
    cueText: result.cue,
    rep: repCount
  };
  const summary = generateRepSummary(result.grade, slots, repCount);

  result.comment = summary;

  recordRepResult(result, now);

  lastCommentTextEl.textContent = `#${repCount} · ${summary}`;
  lastCommentEl.classList.remove("is-hidden");

  const label =
    result.grade === "good" ? "GOOD" :
      result.grade === "fair" ? "FAIR" : "NEEDS WORK";

  showFlash(`${label} · ${result.overall}/100 — ${summary}`, result.grade);

  lastEvaluation = result;
  lastEvaluationAt = now;

  updateProgress();
}

// ------------------------------------------------------------
// CANVAS DRAWING
// ------------------------------------------------------------

function drawLowerBody(landmarks, cw, ch, vw, vh, sx, sy, srcW, srcH, now) {
  if (!landmarks || landmarks.length < 33) return null;

  function getPoint(index) {
    const landmark = landmarks[index];
    if (!isLandmarkVisible(landmark)) {
      markLandmarkMissing(index);
      return null;
    }

    const videoX = landmark.x * vw;
    const videoY = landmark.y * vh;

    const u = (videoX - sx) / srcW;
    const v = (videoY - sy) / srcH;

    if (u < 0 || u > 1 || v < 0 || v > 1) {
      markLandmarkMissing(index);
      return null;
    }

    const rawX = (1 - u) * cw;
    const rawY = v * ch;

    return smoothPoint(index, rawX, rawY);
  }

  const points = {};
  for (const index of LOWER_BODY_INDICES) {
    points[index] = getPoint(index);
  }

  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 4;
  ctx.lineCap = "round";

  for (const [i, j] of LOWER_BODY_CONNECTIONS) {
    const p1 = points[i];
    const p2 = points[j];
    if (!p1 || !p2) continue;

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  for (const index of DRAWN_JOINT_INDICES) {
    const point = points[index];
    if (!point) continue;

    ctx.beginPath();
    ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "#ff3366";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  const measurement = currentExercise.measure(points);

  if (!measurement) {
    leftKneeAngle = null;
    rightKneeAngle = null;
    handlePoseLostFrame();
    setMpStatus("Pose detected; knee landmarks unavailable.");
    return null;
  }

  handlePoseFoundFrame();

  leftKneeAngle = measurement.left;
  rightKneeAngle = measurement.right;

  currentExercise.updateBaseline(points, measurement.avg, now);

  const validation = currentExercise.validate(points, measurement.avg);

  // Cache for NLG slots.
  lastValidation = validation;
  lastHipDropPct = validation.hipDropPct ?? null;

  const lShoulder = points[11], rShoulder = points[12];
  const lHip = points[23], rHip = points[24];
  if (lShoulder && rShoulder && lHip && rHip) {
    const lTilt = calculateTorsoTilt(lShoulder, lHip);
    const rTilt = calculateTorsoTilt(rShoulder, rHip);
    lastTorsoDeg =
      (lTilt !== null && rTilt !== null) ? (lTilt + rTilt) / 2 : null;
  } else {
    lastTorsoDeg = null;
  }

  if (validation.valid) {
    updateRepCounter(measurement.avg, now);
  } else {
    if (repState === "DOWN") repState = "UP";
    resetRepState();
  }

  ctx.font = "bold 18px Arial";
  ctx.fillStyle = "#ffff00";
  ctx.shadowColor = "#000";
  ctx.shadowBlur = 4;

  if (points[25]) {
    ctx.fillText(`${Math.round(leftKneeAngle)}°`, points[25].x + 12, points[25].y);
  }
  if (points[26]) {
    ctx.fillText(`${Math.round(rightKneeAngle)}°`, points[26].x + 12, points[26].y);
  }

  ctx.shadowBlur = 0;

  return {
    points,
    measurement,
    validation,
    avg: measurement.avg
  };
}

// ------------------------------------------------------------
// REP DETECTION
// ------------------------------------------------------------
// repStartTime is captured at DESCENT ONSET (avg drops below upAngle - 10).
// Falls back to DOWN-transition if rep started mid-DOWN.

function updateRepCounter(avgKneeAngle, now) {
  const descentOnsetThreshold = currentExercise.upAngle - 10;

  if (repState === "UP") {
    upSince = null;

    if (avgKneeAngle < descentOnsetThreshold && repStartTime === null) {
      repStartTime = now;
      repStartReason = "descent-onset";
    }

    if (avgKneeAngle < currentExercise.downAngle) {
      if (downSince === null) downSince = now;

      if (now - downSince >= DOWN_HOLD_MS) {
        repState = "DOWN";
        downSince = null;

        if (repStartTime === null) {
          repStartTime = now;
          repStartReason = "down-transition";
        }

        repMinAngle = avgKneeAngle;
        repMaxAngle = avgKneeAngle;
      }
    } else {
      downSince = null;
    }
  } else {
    downSince = null;

    if (repMinAngle === null || avgKneeAngle < repMinAngle) {
      repMinAngle = avgKneeAngle;
    }
    if (repMaxAngle === null || avgKneeAngle > repMaxAngle) {
      repMaxAngle = avgKneeAngle;
    }

    if (repStartTime !== null && now - repStartTime > MAX_REP_DURATION_MS) {
      repState = "UP";
      resetRepState();
      return;
    }

    if (avgKneeAngle > currentExercise.upAngle) {
      if (upSince === null) upSince = now;

      if (now - upSince >= UP_HOLD_MS) {
        repState = "UP";
        repCount++;
        upSince = null;

        const minAngle = repMinAngle;
        const maxAngle = repMaxAngle;
        const durationMs = repStartTime !== null ? now - repStartTime : 0;

        finalizeRepEvaluation(now, { minAngle, maxAngle, durationMs });
        resetRepState();
      }
    } else {
      upSince = null;
    }
  }
}

// ------------------------------------------------------------
// SESSION RECORDING
// ------------------------------------------------------------

function recordFrame(avg, left, right, state, now) {
  if (sessionStartTime === null) sessionStartTime = now;
  if (now - lastFrameLogAt < FRAME_LOG_INTERVAL_MS) return;
  lastFrameLogAt = now;

  sessionLog.push({
    type: "frame",
    t: sessionTimeSec(now).toFixed(3),
    avg: avg === null ? "" : avg.toFixed(2),
    left: left === null ? "" : left.toFixed(2),
    right: right === null ? "" : right.toFixed(2),
    state
  });
}

function recordRepResult(result, now) {
  if (sessionStartTime === null) sessionStartTime = now;

  sessionLog.push({
    type: "rep",
    t: sessionTimeSec(now).toFixed(3),
    rep: repCount,
    minAngle: result.minAngle,
    maxAngle: result.maxAngle,
    durationMs: result.durationMs,
    depthScore: result.depthScore,
    tempoScore: result.tempoScore,
    rangeScore: result.rangeScore,
    overall: result.overall,
    grade: result.grade,
    comment: result.comment || ""
  });

  console.log(`[REP ${repCount}]`, result);
}

// ------------------------------------------------------------
// SESSION EXPORT
// ------------------------------------------------------------

function buildTimestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

function downloadCsv(filename, lines) {
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportRepCountAgreement() {
  const repRows = sessionLog.filter((r) => r.type === "rep");

  if (repRows.length === 0) {
    return { ok: false, reason: "No completed reps to export." };
  }

  const lines = [];

  lines.push(`# exercise,${currentExercise.id}`);
  lines.push(`# exercise_name,"${escapeCsv(currentExercise.name)}"`);
  lines.push(`# session_start_iso,${new Date().toISOString()}`);
  lines.push(`# target_reps,${targetReps}`);
  lines.push(`# auto_rep_count,${repRows.length}`);
  lines.push(`# frame_log_hz,${(1000 / FRAME_LOG_INTERVAL_MS).toFixed(1)}`);
  lines.push(
    `# instructions,"Fill in manual_rep by watching the recorded session. Set match=1 if auto and manual agree on this rep, 0 otherwise."`
  );
  lines.push("");

  lines.push([
    "auto_rep",
    "manual_rep",
    "match",
    "time_s",
    "min_angle_deg",
    "max_angle_deg",
    "range_deg",
    "duration_ms",
    "depth_score",
    "tempo_score",
    "range_score",
    "overall",
    "grade",
    "comment"
  ].join(","));

  for (const r of repRows) {
    const range =
      (r.minAngle !== null && r.maxAngle !== null)
        ? (r.maxAngle - r.minAngle).toFixed(2)
        : "";
    lines.push([
      r.rep,
      "",
      "",
      r.t,
      r.minAngle,
      r.maxAngle,
      range,
      r.durationMs,
      r.depthScore,
      r.tempoScore,
      r.rangeScore,
      r.overall,
      r.grade,
      `"${escapeCsv(r.comment)}"`
    ].join(","));
  }

  const filename =
    `rep_count_agreement_${currentExercise.id}_${buildTimestamp()}.csv`;
  downloadCsv(filename, lines);
  return { ok: true, filename, repCount: repRows.length };
}

function exportKinoveaFrames() {
  const frameRows = sessionLog.filter((r) => r.type === "frame");

  if (frameRows.length === 0) {
    return { ok: false, reason: "No frames logged. Open the camera and move first." };
  }

  const lines = [];

  lines.push(`# exercise,${currentExercise.id}`);
  lines.push(`# exercise_name,"${escapeCsv(currentExercise.name)}"`);
  lines.push(`# session_start_iso,${new Date().toISOString()}`);
  lines.push(`# frame_log_hz,${(1000 / FRAME_LOG_INTERVAL_MS).toFixed(1)}`);
  lines.push(
    `# notes,"Angles are in degrees. Frame rows are downsampled to ${FRAME_LOG_INTERVAL_MS}ms. Blank angles = landmarks unavailable that frame."`
  );
  lines.push("");

  lines.push([
    "frame_index",
    "time_s",
    "state",
    "left_knee_deg",
    "right_knee_deg",
    "avg_knee_deg",
    "left_hip_x",
    "left_hip_y",
    "right_hip_x",
    "right_hip_y",
    "left_knee_x",
    "left_knee_y",
    "right_knee_x",
    "right_knee_y",
    "left_ankle_x",
    "left_ankle_y",
    "right_ankle_x",
    "right_ankle_y"
  ].join(","));

  for (let i = 0; i < frameRows.length; i++) {
    const r = frameRows[i];
    lines.push([
      i,
      r.t,
      r.state,
      r.left,
      r.right,
      r.avg,
      "", "", "", "", "", "", "", "", "", "", "", ""
    ].join(","));
  }

  const filename =
    `kinovea_frames_${currentExercise.id}_${buildTimestamp()}.csv`;
  downloadCsv(filename, lines);
  return { ok: true, filename, frameCount: frameRows.length };
}

function exportSessionCsv() {
  if (sessionLog.length === 0) {
    setStatus("Nothing to export yet.", true);
    return;
  }

  const repResult = exportRepCountAgreement();
  const frameResult = exportKinoveaFrames();

  const parts = [];
  if (repResult.ok) parts.push(`reps: ${repResult.repCount}`);
  if (frameResult.ok) parts.push(`frames: ${frameResult.frameCount}`);

  if (parts.length === 0) {
    setStatus(
      (repResult.reason || frameResult.reason || "Nothing to export."),
      true
    );
    return;
  }

  setStatus(`Exported ${parts.join(", ")}.`);
}

// ------------------------------------------------------------
// VIDEO LOOP
// ------------------------------------------------------------

function drawFrame() {
  if (!stream || !sourceVideo.srcObject) {
    rafId = null;
    return;
  }

  const now = performance.now();

  const vw = sourceVideo.videoWidth;
  const vh = sourceVideo.videoHeight;
  const cw = stageCanvas.width;
  const ch = stageCanvas.height;

  if (vw > 0 && vh > 0) {
    const scale = Math.max(cw / vw, ch / vh);
    const srcW = cw / scale;
    const srcH = ch / scale;
    const sx = (vw - srcW) / 2;
    const sy = (vh - srcH) / 2;

    ctx.save();
    ctx.translate(cw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(sourceVideo, sx, sy, srcW, srcH, 0, 0, cw, ch);
    ctx.restore();

    if (poseLandmarker && sourceVideo.currentTime !== lastVideoTime) {
      lastVideoTime = sourceVideo.currentTime;

      try {
        const results = poseLandmarker.detectForVideo(sourceVideo, now);
        latestLandmarks = results.landmarks?.[0] ?? null;

        if (latestLandmarks) {
          setMpStatus("Pose detected.");
        } else {
          setMpStatus("No pose detected.");
          leftKneeAngle = null;
          rightKneeAngle = null;
          handlePoseLostFrame();
        }
      } catch (error) {
        console.error("Pose detection failed:", error);
        latestLandmarks = null;
        leftKneeAngle = null;
        rightKneeAngle = null;
        setMpStatus("Pose detection error.", true);
      }
    }

    let frameState = null;

    if (latestLandmarks) {
      frameState = drawLowerBody(
        latestLandmarks, cw, ch, vw, vh, sx, sy, srcW, srcH, now
      );

      if (frameState) {
        recordFrame(
          frameState.avg,
          frameState.measurement.left,
          frameState.measurement.right,
          repState,
          now
        );
      }
    }

    const snapshot = snapshotEvaluation(
      frameState ? frameState.avg : null,
      frameState ? frameState.validation : null,
      now
    );
    updateEvaluationPanel(snapshot);

    updatePanel();
    updateFlashVisibility(now);
  }

  rafId = requestAnimationFrame(drawFrame);
}

// ------------------------------------------------------------
// CAMERA CONTROLS
// ------------------------------------------------------------

async function openCamera() {
  if (cameraStarting || stream) return;

  cameraStarting = true;
  setCameraUi(false);

  const session = ++cameraSession;

  setStatus("Loading pose model...");
  setMpStatus("Initializing...");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access is unavailable. Use localhost or HTTPS.");
    }

    await initPoseLandmarker();
    if (session !== cameraSession) return;

    modelReady = true;
    setStatus("Requesting camera access...");

    const newStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    if (session !== cameraSession) {
      newStream.getTracks().forEach(t => t.stop());
      return;
    }

    stream = newStream;
    sourceVideo.srcObject = stream;

    await new Promise((resolve, reject) => {
      if (sourceVideo.videoWidth > 0) { resolve(); return; }
      sourceVideo.onloadedmetadata = resolve;
      sourceVideo.onerror = () => reject(new Error("Could not load the camera video."));
    });

    if (session !== cameraSession) return;

    await sourceVideo.play();
    if (session !== cameraSession) return;

    resetTrackingState();

    setCameraUi(true);
    setStatus("Camera is live.");
    setMpStatus("Searching for pose...");

    drawFrame();
  } catch (error) {
    console.error("Camera initialization failed:", error);

    if (session === cameraSession) {
      let message = error.message || "Could not start the camera.";

      if (error.name === "NotAllowedError") message = "Camera permission denied.";
      else if (error.name === "NotFoundError") message = "No camera device found.";
      else if (error.name === "NotReadableError") message = "Camera is in use by another application.";

      setStatus(message, true);
      closeCamera({ silent: true });
    }
  } finally {
    cameraStarting = false;
    setCameraUi(Boolean(stream));
  }
}

function closeCamera({ silent = false } = {}) {
  cameraSession++;

  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  sourceVideo.onloadedmetadata = null;
  sourceVideo.onerror = null;
  sourceVideo.srcObject = null;

  ctx.clearRect(0, 0, stageCanvas.width, stageCanvas.height);

  resetTrackingState();
  updatePanel();
  updateProgress();

  setCameraUi(false);
  setMpStatus("Waiting.");

  if (!silent) setStatus("Camera is off.");
}

// ------------------------------------------------------------
// EXERCISE SWITCHING
// ------------------------------------------------------------

function switchExercise(id) {
  const next = EXERCISES[id];
  if (!next || next === currentExercise) return;

  currentExercise.reset?.();

  currentExercise = next;

  repCount = 0;
  repState = "UP";
  resetRepState();
  resetTrackingState();

  updatePanel();
  updateProgress();

  if (appSubtitleEl && next.subtitle) {
    appSubtitleEl.textContent = next.subtitle;
  }

  setStatus(`Exercise: ${next.name}`);
}

// ------------------------------------------------------------
// EVENTS
// ------------------------------------------------------------

openBtn.addEventListener("click", openCamera);
closeBtn.addEventListener("click", () => closeCamera());
exportBtn.addEventListener("click", exportSessionCsv);

targetRepsEl.addEventListener("change", () => {
  targetReps = parseInt(targetRepsEl.value, 10) || 10;
  sessionCompleteShown = false;
  updateProgress();
});

if (exerciseSelectEl) {
  exerciseSelectEl.addEventListener("change", () => {
    switchExercise(exerciseSelectEl.value);
  });
}

window.addEventListener("pagehide", () => {
  closeCamera({ silent: true });
  if (poseLandmarker) {
    poseLandmarker.close();
    poseLandmarker = null;
    modelReady = false;
  }
});

// ------------------------------------------------------------
// INIT
// ------------------------------------------------------------

setCameraUi(false);
updatePanel();
updateProgress();
updateEvaluationPanel(null);

if (appSubtitleEl && currentExercise.subtitle) {
  appSubtitleEl.textContent = currentExercise.subtitle;
}