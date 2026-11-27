/* VSL-GH Interactive Dataset Showcase — strictly XSS-safe DOM rendering */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  // Exact anatomical edges matching extract_keypoints.py (25 pose, 70 face, 21 L-hand, 21 R-hand)
  const POSE_EDGES = [
    [9, 10], // shoulders
    [9, 11], [11, 13], // left arm
    [10, 12], [12, 14], // right arm
    [9, 19], [10, 20], [19, 20], // torso
    [0, 9], [0, 10], // head-shoulder reference
  ];
  const HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
  ];

  const state = {
    samples: [],
    corpus: [],
    currentSample: null,
    currentKeypoints: null,
    kpCache: new Map(),
    skelView: "front",
    playing: false,
    speed: 1.0,
    rafId: null,
    layers: { pose: true, face: true, lhand: true, rhand: true },
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function svgEl(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      node.setAttribute(k, String(v));
    }
    return node;
  }

  function splitPillClass(split) {
    if (split === "val") return "pill pill-amber";
    if (split === "test") return "pill pill-indigo";
    return "pill pill-teal";
  }

  function typeLabelVi(stype) {
    if (stype === "Interrogative") return "Nghi vấn (?)";
    if (stype === "Exclamatory") return "Cảm thán (!)";
    return "Trần thuật";
  }

  // ---------------- SAMPLE LIST & PLAYER ----------------
  function renderSampleList() {
    const phaseVal = document.getElementById("filter-phase").value;
    const signerVal = document.getElementById("filter-signer").value;
    const nmmOnly = document.getElementById("filter-nmm-only").checked;

    const filtered = state.samples.filter((s) => {
      if (phaseVal !== "all" && String(s.phase_id) !== phaseVal) return false;
      if (signerVal !== "all" && s.signer_id !== signerVal) return false;
      if (nmmOnly && !s.nmm_note) return false;
      return true;
    });

    const countEl = document.getElementById("sample-count");
    countEl.textContent = "Đang hiện " + filtered.length + " / " + state.samples.length + " mẫu";

    const listEl = document.getElementById("sample-list");
    listEl.replaceChildren();

    if (filtered.length === 0) {
      listEl.appendChild(el("div", "sample-item", "Không có mẫu nào khớp bộ lọc hiện tại."));
      return;
    }

    for (const s of filtered) {
      const btn = el("button", "sample-item" + (state.currentSample && state.currentSample.id === s.id ? " active" : ""));
      btn.type = "button";

      const top = el("div", "sample-item-top");
      top.appendChild(el("span", "", s.sentence_id + " • " + s.signer_id));
      top.appendChild(el("span", splitPillClass(s.split), s.split.toUpperCase()));
      btn.appendChild(top);

      btn.appendChild(el("div", "sample-item-sent", s.vietnamese));

      const meta = el("div", "sample-item-meta");
      meta.appendChild(el("span", "", s.phase_vi));
      meta.appendChild(el("span", "", "• " + s.duration_sec.toFixed(1) + "s"));
      if (s.nmm_note) {
        meta.appendChild(el("span", "pill pill-rose", "NMM: " + s.nmm_note));
      }
      btn.appendChild(meta);

      btn.addEventListener("click", () => selectSample(s));
      listEl.appendChild(btn);
    }

    if (!state.currentSample || !filtered.some((x) => x.id === state.currentSample.id)) {
      selectSample(filtered[0]);
    }
  }

  async function selectSample(sample) {
    pauseBoth();
    state.currentSample = sample;

    // Highlight active item
    renderSampleListHighlight();

    // Update header tags
    const tagsEl = document.getElementById("stage-tags");
    tagsEl.replaceChildren();
    tagsEl.appendChild(el("span", "pill pill-teal", "Giai đoạn " + sample.phase_id + ": " + sample.phase_vi));
    tagsEl.appendChild(el("span", splitPillClass(sample.split), sample.signer_id + " (" + sample.split.toUpperCase() + ")"));
    tagsEl.appendChild(el("span", "pill pill-slate", typeLabelVi(sample.sentence_type)));
    if (sample.nmm_note) {
      tagsEl.appendChild(el("span", "pill pill-rose", "NMM: " + sample.nmm_note));
    }

    document.getElementById("stage-sentence").textContent = sample.vietnamese;
    document.getElementById("stage-highlight").textContent = sample.highlight;

    const metaRight = document.getElementById("stage-meta-right");
    metaRight.replaceChildren();
    const lines = [
      "ID: " + sample.id,
      "Thuận tay: " + sample.handedness,
      "Thời lượng: " + sample.duration_sec.toFixed(2) + "s (" + sample.num_frames + " frames)",
    ];
    for (const ln of lines) {
      metaRight.appendChild(el("div", "", ln));
    }

    // Linguistic cards
    document.getElementById("ling-vi").textContent = sample.vietnamese;
    document.getElementById("ling-hcmc").textContent = sample.hcmc_sign || "—";
    document.getElementById("ling-gloss").textContent = sample.gloss_sequence || "—";
    document.getElementById("raw-elan-text").textContent =
      (sample.raw_elan ? sample.raw_elan.trim() : "(Không có nội dung ELAN)") +
      "\n\n--- Keypoint tensor info ---\n" +
      "Front .npy: (" + sample.num_frames + ", 411) float32  [25 Pose + 70 Face + 42 Hands] × (x, y, z)\n" +
      "Model input: (" + sample.num_frames + ", 822) float32  [411 Position + 411 Velocity]";

    // Videos
    const vf = document.getElementById("video-front");
    const vs = document.getElementById("video-side");
    vf.src = sample.video_front;
    vs.src = sample.video_side;
    vf.playbackRate = state.speed;
    vs.playbackRate = state.speed;
    vf.load();
    vs.load();

    // ELAN timeline
    renderTimeline(sample);

    // Fetch keypoints JSON (cached)
    state.currentKeypoints = null;
    drawEmptySkeleton("Đang tải keypoints 137 điểm...");
    try {
      let kpData = state.kpCache.get(sample.id);
      if (!kpData) {
        const resp = await fetch(sample.keypoints_json);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        kpData = await resp.json();
        state.kpCache.set(sample.id, kpData);
      }
      if (state.currentSample && state.currentSample.id === sample.id) {
        state.currentKeypoints = kpData;
        updateVisualsAtTime(0);
      }
    } catch (_err) {
      drawEmptySkeleton("Không tải được keypoints cho mẫu này.");
    }
  }

  function renderSampleListHighlight() {
    const items = document.querySelectorAll("#sample-list .sample-item");
    const phaseVal = document.getElementById("filter-phase").value;
    const signerVal = document.getElementById("filter-signer").value;
    const nmmOnly = document.getElementById("filter-nmm-only").checked;
    const filtered = state.samples.filter((s) => {
      if (phaseVal !== "all" && String(s.phase_id) !== phaseVal) return false;
      if (signerVal !== "all" && s.signer_id !== signerVal) return false;
      if (nmmOnly && !s.nmm_note) return false;
      return true;
    });
    items.forEach((btn, idx) => {
      const s = filtered[idx];
      if (s && state.currentSample && s.id === state.currentSample.id) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  function renderTimeline(sample) {
    const track = document.getElementById("timeline-track");
    const ruler = document.getElementById("timeline-ruler");
    track.replaceChildren();
    ruler.replaceChildren();

    const totalDur = Math.max(
      sample.duration_sec,
      sample.timeline.reduce((m, g) => Math.max(m, g.end_sec), 0.1)
    );

    const playhead = el("div", "timeline-playhead");
    playhead.id = "timeline-playhead";
    playhead.style.left = "0%";
    track.appendChild(playhead);

    sample.timeline.forEach((seg, idx) => {
      const leftPct = Math.max(0, Math.min(100, (seg.start_sec / totalDur) * 100));
      const widthPct = Math.max(2, Math.min(100 - leftPct, ((seg.end_sec - seg.start_sec) / totalDur) * 100));
      const b = el("button", "gloss-seg", seg.gloss);
      b.type = "button";
      b.dataset.idx = String(idx);
      b.style.left = leftPct.toFixed(2) + "%";
      b.style.width = widthPct.toFixed(2) + "%";
      b.title = seg.gloss + " (" + seg.start_sec.toFixed(2) + "s – " + seg.end_sec.toFixed(2) + "s | frame " + seg.start_frame + "–" + seg.end_frame + ")";
      b.addEventListener("click", () => {
        seekBoth(seg.start_sec);
      });
      track.appendChild(b);
    });

    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const sec = (totalDur * i) / ticks;
      ruler.appendChild(el("span", "", sec.toFixed(2) + "s"));
    }
  }

  // ---------------- SYNCHRONIZED PLAYBACK & SKELETON RENDERER ----------------
  function getDuration() {
    const vf = document.getElementById("video-front");
    if (vf && Number.isFinite(vf.duration) && vf.duration > 0) return vf.duration;
    return state.currentSample ? state.currentSample.duration_sec : 1;
  }

  function playBoth() {
    const vf = document.getElementById("video-front");
    const vs = document.getElementById("video-side");
    if (!vf.src) return;
    if (vf.ended) {
      vf.currentTime = 0;
      vs.currentTime = 0;
    }
    vf.playbackRate = state.speed;
    vs.playbackRate = state.speed;
    vf.play().catch(() => {});
    vs.play().catch(() => {});
    state.playing = true;
    document.getElementById("btn-play").textContent = "❚❚ Tạm dừng";
    startSyncLoop();
  }

  function pauseBoth() {
    const vf = document.getElementById("video-front");
    const vs = document.getElementById("video-side");
    vf.pause();
    vs.pause();
    state.playing = false;
    document.getElementById("btn-play").textContent = "▶ Phát đồng bộ";
    if (state.rafId !== null) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
  }

  function seekBoth(sec) {
    const vf = document.getElementById("video-front");
    const vs = document.getElementById("video-side");
    const dur = getDuration();
    const clamped = Math.max(0, Math.min(dur, sec));
    vf.currentTime = clamped;
    if (Number.isFinite(vs.duration)) {
      vs.currentTime = Math.min(vs.duration, clamped);
    } else {
      vs.currentTime = clamped;
    }
    updateVisualsAtTime(clamped);
  }

  function startSyncLoop() {
    if (state.rafId !== null) cancelAnimationFrame(state.rafId);
    const vf = document.getElementById("video-front");
    const vs = document.getElementById("video-side");

    function tick() {
      if (!state.playing) return;
      const t = vf.currentTime || 0;
      if (Number.isFinite(vs.duration) && Math.abs(vs.currentTime - t) > 0.12 && t < vs.duration) {
        vs.currentTime = t;
      }
      updateVisualsAtTime(t);
      if (vf.ended) {
        pauseBoth();
        return;
      }
      state.rafId = requestAnimationFrame(tick);
    }
    state.rafId = requestAnimationFrame(tick);
  }

  function updateVisualsAtTime(sec) {
    const sample = state.currentSample;
    if (!sample) return;
    const dur = getDuration();
    const ratio = dur > 0 ? Math.max(0, Math.min(1, sec / dur)) : 0;

    // Scrubber & playhead
    document.getElementById("scrubber").value = String(Math.round(ratio * 1000));
    const playhead = document.getElementById("timeline-playhead");
    if (playhead) playhead.style.left = (ratio * 100).toFixed(2) + "%";

    // Frame index
    const framesArr = state.currentKeypoints
      ? (state.skelView === "side" ? state.currentKeypoints.side : state.currentKeypoints.front)
      : null;
    const totalFrames = framesArr ? framesArr.length : sample.num_frames;
    const frameIdx = Math.max(0, Math.min(totalFrames - 1, Math.floor(sec * 30)));
    document.getElementById("time-readout").textContent =
      sec.toFixed(2) + "s / " + dur.toFixed(2) + "s • Frame " + frameIdx + " / " + totalFrames;

    // Active gloss highlight
    let activeGloss = "";
    const segButtons = document.querySelectorAll("#timeline-track .gloss-seg");
    sample.timeline.forEach((seg, idx) => {
      const isAct = sec >= seg.start_sec && sec <= seg.end_sec;
      if (isAct) activeGloss = seg.gloss;
      if (segButtons[idx]) {
        segButtons[idx].classList.toggle("active", isAct);
      }
    });
    const overlay = document.getElementById("canvas-active-gloss");
    overlay.textContent = activeGloss ? "GLOSS: " + activeGloss : "—";

    // Draw skeleton
    if (framesArr && framesArr[frameIdx]) {
      drawSkeletonFrame(framesArr[frameIdx]);
    }
  }

  function drawEmptySkeleton(msg) {
    const canvas = document.getElementById("skeleton-canvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "15px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
  }

  function drawSkeletonFrame(flatPts) {
    const canvas = document.getElementById("skeleton-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Pure black background matching paper Fig. 1(b)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);

    function pt(i, allowExtend) {
      const x = flatPts[i * 2];
      const y = flatPts[i * 2 + 1];
      if (!x && !y) return null;
      const hi = allowExtend ? 1.45 : 1.15;
      const lo = allowExtend ? -0.35 : -0.15;
      if (x < lo || x > hi || y < lo || y > hi) return null;
      return [x * W, y * H];
    }

    function mid(a, b) {
      if (!a || !b) return null;
      return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    }

    function line(a, b, color, width) {
      if (!a || !b) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }

    function dot(p, color, r) {
      if (!p) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    }

    // 1. OpenPose-style Body Skeleton (light steel-blue #7ec8e3, central spine + neck + legs)
    if (state.layers.pose) {
      // Auto-detect pose layout: *_F.npy uses custom 25-pt subset (shoulders=9,10; hips=19,20),
      // while *_S.npy uses raw MediaPipe 0..24 (mouth=9,10; shoulders=11,12; elbows=13,14; wrists=15,16; hips=23,24).
      const p0 = pt(0, false), p9 = pt(9, false), p11 = pt(11, false);
      const isRawMP25 = (state.skelView === "side") || (p0 && p9 && p11 && Math.abs(p9[1] - p0[1]) < 0.08 * H && (p11[1] - p9[1]) > 0.05 * H);

      const lSh  = isRawMP25 ? pt(11, false) : pt(9, false);
      const rSh  = isRawMP25 ? pt(12, false) : pt(10, false);
      const lEl  = isRawMP25 ? pt(13, false) : pt(11, false);
      const rEl  = isRawMP25 ? pt(14, false) : pt(12, false);
      const lWr  = isRawMP25 ? pt(15, false) : pt(13, false);
      const rWr  = isRawMP25 ? pt(16, false) : pt(14, false);
      const lHip = isRawMP25 ? pt(23, true)  : pt(19, true);
      const rHip = isRawMP25 ? pt(24, true)  : pt(20, true);
      const lKn  = isRawMP25 ? null          : pt(21, true);
      const rKn  = isRawMP25 ? null          : pt(22, true);
      const lAnk = isRawMP25 ? null          : pt(23, true);
      const rAnk = isRawMP25 ? null          : pt(24, true);

      const neck = mid(lSh, rSh);
      const midHip = mid(lHip, rHip);
      // In side profile, connect neck to lower jaw/ear-base (landmark ~90) like paper Fig. 1(b); in front view use chin (94)
      const chin = isRawMP25 ? (pt(90, false) || pt(94, false)) : pt(94, false);
      const headBase = chin || (p0 && neck ? [p0[0], p0[1] + 24] : null);

      const boneColor = "#7ec8e3";
      const jointColor = "#93d5ed";
      const bw = 2.6;

      // Neck & central spine (matching paper Fig. 1b)
      line(headBase, neck, boneColor, bw);
      line(neck, midHip, boneColor, bw);

      // Shoulders & arms
      line(neck, lSh, boneColor, bw);
      line(neck, rSh, boneColor, bw);
      line(lSh, lEl, boneColor, bw);
      line(lEl, lWr, boneColor, bw);
      line(rSh, rEl, boneColor, bw);
      line(rEl, rWr, boneColor, bw);

      // Pelvis & seated legs
      line(midHip, lHip, boneColor, bw);
      line(midHip, rHip, boneColor, bw);
      line(lHip, lKn, boneColor, bw);
      line(rHip, rKn, boneColor, bw);
      line(lKn, lAnk, boneColor, bw);
      line(rKn, rAnk, boneColor, bw);

      // Clean anatomical joints only (no stray dots on face/ears)
      const joints = [headBase, neck, midHip, lSh, rSh, lEl, rEl, lWr, rWr, lHip, rHip, lKn, rKn, lAnk, rAnk];
      for (const j of joints) {
        dot(j, jointColor, 3.8);
      }
    }

    // 2. Face 70 landmarks in warm orange (#fb923c / #f97316) matching paper Fig. 1(b)
    if (state.layers.face) {
      const faceDot = "#fb923c";
      const faceLine = "rgba(251, 146, 60, 0.55)";

      function drawChain(startIdx, endIdx, closed) {
        ctx.strokeStyle = faceLine;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        let started = false;
        let firstPt = null;
        for (let i = startIdx; i <= endIdx; i++) {
          const p = pt(i, false);
          if (!p) continue;
          if (!started) {
            ctx.moveTo(p[0], p[1]);
            firstPt = p;
            started = true;
          } else {
            ctx.lineTo(p[0], p[1]);
          }
        }
        if (closed && started && firstPt) ctx.lineTo(firstPt[0], firstPt[1]);
        if (started) ctx.stroke();
      }

      // Delicate facial contours (lips, eyes, eyebrows, nose, jawline)
      drawChain(25, 34, false); // upper/outer lip half
      drawChain(35, 44, false); // lower/inner lip half
      drawChain(45, 50, true);  // left eye
      drawChain(51, 56, true);  // right eye
      drawChain(57, 61, false); // left eyebrow
      drawChain(62, 66, false); // right eyebrow
      drawChain(67, 75, false); // nose bridge/base
      drawChain(76, 94, false); // extracted face contour (forehead -> cheek -> chin)

      // All 70 extracted facial landmark dots
      for (let i = 25; i < 95; i++) {
        dot(pt(i, false), faceDot, 1.8);
      }

      // Complete symmetric left cheek in Frontal view so the oval outline matches paper Fig. 1(b)
      if (state.skelView === "front") {
        const topP = pt(76, false);
        const chinP = pt(94, false);
        if (topP && chinP) {
          const ax = chinP[0] - topP[0];
          const ay = chinP[1] - topP[1];
          const len2 = ax * ax + ay * ay;
          if (len2 > 100) {
            const mirrored = [];
            for (let i = 76; i <= 94; i++) {
              const p = pt(i, false);
              if (!p) continue;
              const t = ((p[0] - topP[0]) * ax + (p[1] - topP[1]) * ay) / len2;
              const px = topP[0] + t * ax;
              const py = topP[1] + t * ay;
              const mx = 2 * px - p[0];
              const my = 2 * py - p[1];
              mirrored.push([mx, my]);
            }
            if (mirrored.length > 2) {
              ctx.strokeStyle = faceLine;
              ctx.lineWidth = 1.1;
              ctx.beginPath();
              ctx.moveTo(mirrored[0][0], mirrored[0][1]);
              for (let k = 1; k < mirrored.length; k++) ctx.lineTo(mirrored[k][0], mirrored[k][1]);
              ctx.stroke();
              for (let k = 1; k < mirrored.length - 1; k++) {
                dot(mirrored[k], faceDot, 1.8);
              }
            }
          }
        }
      }
    }

    // 3. Left Hand (95..115) — green matching paper Fig. 1(b)
    if (state.layers.lhand) {
      drawHand(ctx, pt, 95, "#22c55e", "#4ade80");
    }

    // 4. Right Hand (116..136) — purple matching paper Fig. 1(b)
    if (state.layers.rhand) {
      drawHand(ctx, pt, 116, "#9333ea", "#c084fc");
    }
  }

  function drawHand(ctx, ptFn, offset, strokeColor, fillColor) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.9;
    ctx.lineCap = "round";
    for (const [a, b] of HAND_EDGES) {
      const pa = ptFn(offset + a, true), pb = ptFn(offset + b, true);
      if (pa && pb) {
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      }
    }
    ctx.fillStyle = fillColor;
    for (let i = 0; i < 21; i++) {
      const p = ptFn(offset + i, true);
      if (p) {
        ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  // ---------------- PURE SVG CHARTS ----------------
  function renderPhaseChart() {
    const data = [
      { label: "Khám lâm sàng (Clinical)", count: 130, nmm: 62, color: "#0d9488" },
      { label: "Đăng ký khám (Registration)", count: 68, nmm: 14, color: "#4f46e5" },
      { label: "Cận lâm sàng (Paraclinical)", count: 40, nmm: 0, color: "#0284c7" },
      { label: "Khám thể lực (Physical)", count: 32, nmm: 3, color: "#d97706" },
      { label: "Kết luận (Conclusion)", count: 30, nmm: 3, color: "#e11d48" },
    ];
    const host = document.getElementById("chart-phases");
    host.replaceChildren();
    const W = 560, H = 250;
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Biểu đồ phân bố 5 giai đoạn khám" });

    data.forEach((d, i) => {
      const y = 18 + i * 44;
      const barW = (d.count / 130) * 280;
      const nmmW = (d.nmm / 130) * 280;

      const lbl = svgEl("text", { x: 0, y: y + 17, fill: "#0f172a", "font-size": 12.5, "font-weight": 600 });
      lbl.textContent = d.label;
      svg.appendChild(lbl);

      svg.appendChild(svgEl("rect", { x: 205, y: y + 2, width: 280, height: 22, rx: 6, fill: "#e2e8f0" }));
      svg.appendChild(svgEl("rect", { x: 205, y: y + 2, width: barW.toFixed(1), height: 22, rx: 6, fill: d.color }));
      if (nmmW > 0) {
        svg.appendChild(svgEl("rect", { x: 205, y: y + 17, width: nmmW.toFixed(1), height: 7, rx: 3, fill: "rgba(255,255,255,0.65)" }));
      }

      const val = svgEl("text", { x: 493, y: y + 17, fill: "#334155", "font-size": 12, "font-weight": 700 });
      val.textContent = d.count + " (" + ((d.count / 300) * 100).toFixed(1) + "%)";
      svg.appendChild(val);
    });

    const note = svgEl("text", { x: 205, y: H - 6, fill: "#64748b", "font-size": 11 });
    note.textContent = "Dải trắng bên trong thanh thể hiện số câu có ghi chú NMM (tổng 82/300 câu).";
    svg.appendChild(note);
    host.appendChild(svg);
  }

  function renderTypesNmmChart() {
    const rows = [
      { group: "Loại câu", label: "Trần thuật (Declarative)", count: 175, max: 300, color: "#0d9488" },
      { group: "Loại câu", label: "Nghi vấn (Interrogative)", count: 109, max: 300, color: "#4f46e5" },
      { group: "Loại câu", label: "Cảm thán (Exclamatory)", count: 16, max: 300, color: "#e11d48" },
      { group: "NMM", label: "Biểu cảm buồn / đau (Sad)", count: 47, max: 83, color: "#0284c7" },
      { group: "NMM", label: "Gật đầu (Nod) & Nhăn mặt (Frown)", count: 24, max: 83, color: "#d97706" },
      { group: "NMM", label: "Vui / Lắc đầu / Khác", count: 12, max: 83, color: "#16a34a" },
    ];
    const host = document.getElementById("chart-types-nmm");
    host.replaceChildren();
    const W = 560, H = 250;
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Phân bố loại câu và NMM" });

    rows.forEach((d, i) => {
      const y = 12 + i * 38;
      const barW = (d.count / d.max) * 240;

      const lbl = svgEl("text", { x: 0, y: y + 16, fill: "#0f172a", "font-size": 12, "font-weight": 600 });
      lbl.textContent = d.label;
      svg.appendChild(lbl);

      svg.appendChild(svgEl("rect", { x: 235, y: y + 3, width: 240, height: 18, rx: 5, fill: "#e2e8f0" }));
      svg.appendChild(svgEl("rect", { x: 235, y: y + 3, width: barW.toFixed(1), height: 18, rx: 5, fill: d.color }));

      const val = svgEl("text", { x: 483, y: y + 16, fill: "#334155", "font-size": 11.5, "font-weight": 700 });
      val.textContent = d.count + " (" + ((d.count / d.max) * 100).toFixed(1) + "%)";
      svg.appendChild(val);
    });
    host.appendChild(svg);
  }

  // ---------------- 300-SENTENCE CORPUS BROWSER ----------------
  function renderCorpusTable() {
    const q = (document.getElementById("corpus-search").value || "").trim().toLowerCase();
    const ph = document.getElementById("corpus-phase").value;
    const st = document.getElementById("corpus-type").value;
    const nmmOnly = document.getElementById("corpus-nmm-only").checked;

    const filtered = state.corpus.filter((c) => {
      if (ph !== "all" && String(c.phase_id) !== ph) return false;
      if (st !== "all" && c.sentence_type !== st) return false;
      if (nmmOnly && !c.nmm_note) return false;
      if (q) {
        const hay = (c.sentence_id + " " + c.vietnamese + " " + c.hcmc_sign + " " + c.gloss_sequence).toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    document.getElementById("corpus-count").textContent = "Hiển thị " + filtered.length + " / " + state.corpus.length + " câu";
    const tbody = document.getElementById("corpus-tbody");
    tbody.replaceChildren();

    const frag = document.createDocumentFragment();
    for (const c of filtered) {
      const tr = document.createElement("tr");
      tr.appendChild(el("td", "mono", c.sentence_id));
      tr.appendChild(el("td", "", c.phase_vi));
      tr.appendChild(el("td", "", c.vietnamese));

      const tdHcmc = document.createElement("td");
      tdHcmc.appendChild(document.createTextNode(c.hcmc_sign || "—"));
      if (c.nmm_note) {
        tdHcmc.appendChild(document.createTextNode(" "));
        tdHcmc.appendChild(el("span", "pill pill-rose", c.nmm_note));
      }
      tr.appendChild(tdHcmc);

      tr.appendChild(el("td", "mono", c.gloss_sequence));
      tr.appendChild(el("td", "", typeLabelVi(c.sentence_type)));
      frag.appendChild(tr);
    }
    tbody.appendChild(frag);
  }

  // ---------------- BENCHMARK TABLES (TABLES 6, 7, 8, 10) ----------------
  const BENCHMARKS = {
    t6: {
      title: "Table 6 — Architecture Benchmark trên VSL-GH (Decoder 6 lớp Transformer chung, Supervised Frame CE w=0.3)",
      headers: ["Mô hình Encoder", "Params (M)", "FPS", "BLEU-4 ↑", "WER (%) ↓", "chrF ↑", "ROUGE-L ↑"],
      rows: [
        { cells: ["Transformer (Vaswani et al., 2017)", "45.2", "85", "70.65 ± 2.05", "28.84 ± 1.95", "74.42 ± 0.99", "79.38 ± 1.12"], bleu: 70.65 },
        { cells: ["ST-GCN + Transformer (Yan et al., 2018)", "21.5", "110", "71.75 ± 0.98", "27.91 ± 0.89", "75.58 ± 1.35", "80.81 ± 0.94"], bleu: 71.75 },
        { cells: ["Mamba (Gu & Dao, 2023)", "18.4", "135", "79.41 ± 2.14", "20.76 ± 1.68", "81.60 ± 2.12", "85.37 ± 1.40"], bleu: 79.41 },
        { cells: ["Conformer (Gulati et al., 2020)", "38.6", "95", "81.92 ± 0.62", "17.78 ± 0.68", "83.99 ± 0.57", "87.43 ± 0.42"], bleu: 81.92 },
        { cells: ["TCN (Bai et al., 2018) — Nhẹ & nhanh nhất", "12.3", "150", "82.60 ± 0.31", "17.11 ± 0.65", "83.95 ± 0.27", "87.49 ± 0.44"], bleu: 82.60, hi: true },
      ],
      note: "TCN đạt điểm BLEU-4 cao nhất (82.60 ± 0.31) trong khi chỉ tốn 12.3M tham số và đạt tốc độ suy luận thời gian thực 150 FPS.",
    },
    t7: {
      title: "Table 7 — Nghiên cứu Triệt tiêu (Ablation Studies) trên TCN Encoder",
      headers: ["Nhóm thí nghiệm", "Cấu hình", "BLEU-4 ↑", "WER (%) ↓", "chrF ↑", "ROUGE-L ↑"],
      rows: [
        { cells: ["(a) Giám sát Gloss", "CTC Only (ctc=0.5, fce=0.0)", "67.85 ± 2.03", "29.53 ± 3.25", "71.66 ± 2.61", "77.43 ± 1.70"], bleu: 67.85 },
        { cells: ["(a) Giám sát Gloss", "Gloss-Free (0.0 / 0.0)", "79.24 ± 0.96", "21.11 ± 1.42", "81.41 ± 0.91", "84.92 ± 1.17"], bleu: 79.24 },
        { cells: ["(a) Giám sát Gloss", "CTC + Supervised (0.3 / 0.3)", "78.92 ± 2.44", "20.04 ± 2.69", "80.64 ± 2.28", "84.86 ± 1.86"], bleu: 78.92 },
        { cells: ["(a) Giám sát Gloss", "Supervised Frame CE (0.0 / 0.3)", "82.60 ± 0.31", "17.11 ± 0.65", "83.95 ± 0.27", "87.49 ± 0.44"], bleu: 82.60, hi: true },
        { cells: ["(b) Đặc trưng & Góc quay", "w/o Face (chỉ Pose & Hands)", "76.69 ± 1.19", "23.60 ± 1.30", "78.59 ± 0.86", "82.52 ± 0.80"], bleu: 76.69 },
        { cells: ["(b) Đặc trưng & Góc quay", "w/o Velocity (bỏ vận tốc Δt)", "81.42 ± 0.47", "18.50 ± 0.55", "82.95 ± 0.45", "86.71 ± 0.35"], bleu: 81.42 },
        { cells: ["(b) Đặc trưng & Góc quay", "Multi-view (Front + Side)", "81.32 ± 1.73", "18.08 ± 1.57", "82.95 ± 1.01", "86.80 ± 1.04"], bleu: 81.32 },
        { cells: ["(b) Đặc trưng & Góc quay", "Baseline: Front + Face + Vel.", "82.60 ± 0.31", "17.11 ± 0.65", "83.95 ± 0.27", "87.49 ± 0.44"], bleu: 82.60, hi: true },
      ],
      note: "Bỏ đặc trưng khuôn mặt (70 điểm NMM) làm giảm sâu nhất (−5.91 BLEU-4), khẳng định vai trò ngữ pháp & ngữ nghĩa của biểu cảm phi thủ công trong NNKH.",
    },
    t8: {
      title: "Table 8 — Kiểm chứng chéo Leave-One-Subject-Out (LOSO 6-Fold, TCN Supervised)",
      headers: ["Signer Test (Hold-out)", "Tập Train", "BLEU-4 ↑", "WER (%) ↓", "chrF ↑", "ROUGE-L ↑"],
      rows: [
        { cells: ["S01", "6,600 video", "80.34 ± 2.93", "19.07 ± 2.28", "81.76 ± 2.63", "85.82 ± 1.80"], bleu: 80.34 },
        { cells: ["S02", "6,600 video", "80.01 ± 1.70", "19.40 ± 1.48", "81.74 ± 1.74", "85.70 ± 1.24"], bleu: 80.01 },
        { cells: ["S03 (Tay trái)", "6,600 video", "81.12 ± 1.53", "19.10 ± 1.36", "82.43 ± 1.66", "86.38 ± 0.95"], bleu: 81.12 },
        { cells: ["S04", "6,600 video", "79.14 ± 2.66", "20.49 ± 2.17", "81.15 ± 2.19", "85.47 ± 1.29"], bleu: 79.14 },
        { cells: ["S05", "7,800 video", "80.92 ± 2.62", "18.54 ± 1.93", "82.51 ± 2.12", "86.34 ± 1.60"], bleu: 80.92 },
        { cells: ["S06", "7,800 video", "82.83 ± 1.30", "17.02 ± 1.03", "84.13 ± 1.27", "87.66 ± 0.83"], bleu: 82.83 },
        { cells: ["Trung bình 6 Fold (Avg.)", "—", "80.73 ± 1.96", "18.94 ± 1.71", "82.29 ± 1.94", "86.23 ± 1.29"], bleu: 80.73, hi: true },
      ],
      note: "Dù giữ riêng từng người ký hiệu (kể cả S03 thuận tay trái), TCN vẫn duy trì trung bình 80.73 BLEU-4 ổn định trên cả 6 fold.",
    },
    t10: {
      title: "Table 10 — Kiểm chứng Mở rộng Dữ liệu Tổng hợp 5× (Synthetic Data Scaling + Fine-tuning, S06 Test, beam=4)",
      headers: ["Chiến lược huấn luyện", "Quy mô tập Train", "BLEU-4 ↑", "WER (%) ↓", "chrF ↑", "ROUGE-L ↑"],
      rows: [
        { cells: ["Baseline (VSL-GH gốc)", "3,600 clip (1×)", "82.60 ± 0.31", "17.11 ± 0.65", "83.95 ± 0.27", "87.49 ± 0.44"], bleu: 82.60 },
        { cells: ["Synthetic 5× Pre-train + Fine-tune", "18,000 clip (5× trên S01–S04)", "84.75 ± 1.46", "15.84 ± 1.61", "85.12 ± 1.35", "88.75 ± 1.12"], bleu: 84.75, hi: true },
      ],
      note: "Nhân bản tăng cường 5× offline chỉ trên tập train (S01–S04), giữ nguyên tập test thật S06 giúp cải thiện đồng đều cả 4 chỉ số (+2.15 BLEU-4, −1.27% WER).",
    },
  };

  function renderBenchmarkTab(key) {
    const spec = BENCHMARKS[key];
    const panel = document.getElementById("benchmark-panel");
    panel.replaceChildren();
    panel.appendChild(el("h3", "", spec.title));

    const wrap = el("div", "table-scroll");
    const tbl = document.createElement("table");
    const thead = document.createElement("thead");
    const htr = document.createElement("tr");
    for (const h of spec.headers) htr.appendChild(el("th", "", h));
    thead.appendChild(htr);
    tbl.appendChild(thead);

    const tbody = document.createElement("tbody");
    const bleuColIdx = spec.headers.findIndex((h) => h.startsWith("BLEU-4"));
    for (const r of spec.rows) {
      const tr = document.createElement("tr");
      if (r.hi) tr.className = "row-highlight";
      r.cells.forEach((cellVal, idx) => {
        const td = document.createElement("td");
        if (idx === bleuColIdx && typeof r.bleu === "number") {
          const mw = el("div", "metric-bar-wrap");
          const bg = el("div", "metric-bar-bg");
          const fill = el("div", "metric-bar-fill");
          fill.style.width = Math.max(0, Math.min(100, r.bleu)).toFixed(1) + "%";
          bg.appendChild(fill);
          mw.appendChild(bg);
          mw.appendChild(el(r.hi ? "strong" : "span", "", cellVal));
          td.appendChild(mw);
        } else {
          td.appendChild(el(r.hi ? "strong" : "span", "", cellVal));
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    tbl.appendChild(tbody);
    wrap.appendChild(tbl);
    panel.appendChild(wrap);

    const noteP = el("p", "split-desc", spec.note);
    noteP.style.marginTop = "12px";
    noteP.style.marginBottom = "0";
    panel.appendChild(noteP);
  }

  // ---------------- WIRE EVENTS & INIT ----------------
  function bindEvents() {
    document.getElementById("filter-phase").addEventListener("change", renderSampleList);
    document.getElementById("filter-signer").addEventListener("change", renderSampleList);
    document.getElementById("filter-nmm-only").addEventListener("change", renderSampleList);

    document.getElementById("btn-play").addEventListener("click", () => {
      if (state.playing) pauseBoth();
      else playBoth();
    });
    document.getElementById("btn-restart").addEventListener("click", () => {
      seekBoth(0);
      playBoth();
    });
    document.getElementById("scrubber").addEventListener("input", (e) => {
      const dur = getDuration();
      const frac = Number(e.target.value) / 1000;
      seekBoth(dur * frac);
    });

    document.querySelectorAll(".speed-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.speed = Number(btn.dataset.speed) || 1.0;
        document.querySelectorAll(".speed-btn").forEach((b) => b.classList.toggle("active", b === btn));
        document.getElementById("video-front").playbackRate = state.speed;
        document.getElementById("video-side").playbackRate = state.speed;
      });
    });

    const btnF = document.getElementById("btn-skel-front");
    const btnS = document.getElementById("btn-skel-side");
    btnF.addEventListener("click", () => {
      state.skelView = "front";
      btnF.classList.add("active");
      btnS.classList.remove("active");
      updateVisualsAtTime(document.getElementById("video-front").currentTime || 0);
    });
    btnS.addEventListener("click", () => {
      state.skelView = "side";
      btnS.classList.add("active");
      btnF.classList.remove("active");
      updateVisualsAtTime(document.getElementById("video-front").currentTime || 0);
    });

    for (const [id, key] of [
      ["toggle-pose", "pose"],
      ["toggle-face", "face"],
      ["toggle-lhand", "lhand"],
      ["toggle-rhand", "rhand"],
    ]) {
      document.getElementById(id).addEventListener("change", (e) => {
        state.layers[key] = e.target.checked;
        updateVisualsAtTime(document.getElementById("video-front").currentTime || 0);
      });
    }

    document.getElementById("corpus-search").addEventListener("input", renderCorpusTable);
    document.getElementById("corpus-phase").addEventListener("change", renderCorpusTable);
    document.getElementById("corpus-type").addEventListener("change", renderCorpusTable);
    document.getElementById("corpus-nmm-only").addEventListener("change", renderCorpusTable);

    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => {
          const isAct = b === btn;
          b.classList.toggle("active", isAct);
          b.setAttribute("aria-selected", String(isAct));
        });
        renderBenchmarkTab(btn.dataset.tab);
      });
    });

    document.getElementById("video-front").addEventListener("loadedmetadata", () => {
      updateVisualsAtTime(0);
    });
  }

  async function init() {
    bindEvents();
    renderPhaseChart();
    renderTypesNmmChart();
    renderBenchmarkTab("t6");

    try {
      const [sResp, cResp] = await Promise.all([
        fetch("data/samples.json"),
        fetch("data/corpus_300.json"),
      ]);
      state.samples = await sResp.json();
      state.corpus = await cResp.json();
      renderSampleList();
      renderCorpusTable();
    } catch (_err) {
      document.getElementById("stage-sentence").textContent =
        "Không thể tải dữ liệu JSON. Vui lòng mở trang thông qua máy chủ cục bộ (python serve.py).";
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
