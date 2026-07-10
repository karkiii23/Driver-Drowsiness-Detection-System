import cv2
import gradio as gr

from config import PAGE_TITLE

from modules.drowsiness import DrowsinessDetector


# ---------------------------------------
# Telemetry rendering
# ---------------------------------------
# The live status readout is built as an HTML "instrument strip"
# (chips + a status pill) instead of a plain textbox, and re-rendered
# on every frame from the detector's current readings.
# ---------------------------------------

def _is_alert(status_text: str) -> bool:
    s = (status_text or "").lower()
    return any(k in s for k in ("drowsy", "alert", "closed", "sleep"))


def build_telemetry_html(detector: DrowsinessDetector | None) -> str:

    if detector is None:
        return """
        <div class="telemetry-row">
            <div class="chip"><span class="chip-label">EAR</span><span class="chip-value">--.--</span></div>
            <div class="chip"><span class="chip-label">MAR</span><span class="chip-value">--.--</span></div>
            <div class="chip"><span class="chip-label">Blinks</span><span class="chip-value">--</span></div>
            <div class="chip"><span class="chip-label">Yawns</span><span class="chip-value">--</span></div>
            <div class="status-pill status-idle"><span class="status-dot"></span>Waiting for feed</div>
        </div>
        """

    alert = _is_alert(detector.status)
    pill_class = "status-alert" if alert else "status-safe"

    return f"""
    <div class="telemetry-row">
        <div class="chip"><span class="chip-label">EAR</span><span class="chip-value">{detector.ear:.2f}</span></div>
        <div class="chip"><span class="chip-label">MAR</span><span class="chip-value">{detector.mar:.2f}</span></div>
        <div class="chip"><span class="chip-label">Blinks</span><span class="chip-value">{detector.blink_count}</span></div>
        <div class="chip"><span class="chip-label">Yawns</span><span class="chip-value">{detector.yawn_count}</span></div>
        <div class="status-pill {pill_class}"><span class="status-dot"></span>{detector.status}</div>
    </div>
    """


# ---------------------------------------
# Live Webcam Streaming (real-time, in-browser)
# ---------------------------------------
# The browser captures the user's own webcam and sends one frame at
# a time to this function; each frame is annotated and streamed
# straight back. Nothing is written to disk here — this is a live
# preview. A fresh DrowsinessDetector is created per browser session
# (via gr.State) so blink/yawn counters aren't shared between users.
# ---------------------------------------

def live_frame_handler(frame, detector):

    if frame is None:
        return None, detector, build_telemetry_html(None)

    if detector is None:
        detector = DrowsinessDetector()

    # Gradio's webcam Image component delivers RGB frames; the
    # detector's drawing/mediapipe pipeline expects BGR (OpenCV
    # convention), so convert both ways.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr = detector.detect(frame_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb, detector, build_telemetry_html(detector)


def reset_live_session():
    return None, build_telemetry_html(None)


# ---------------------------------------
# Styling
# ---------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --dash-bg: #0B0F14;
    --dash-panel: #141B22;
    --dash-panel-2: #101720;
    --dash-line: #232C36;
    --dash-text: #E8EDF2;
    --dash-text-dim: #8494A3;
    --dash-safe: #33D6A6;
    --dash-safe-dim: #1E4B3D;
    --dash-alert: #FF6B4A;
    --dash-alert-dim: #5A2A1E;
}

.gradio-container {
    background: var(--dash-bg) !important;
    color: var(--dash-text) !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 1180px !important;
}

/* ---------- Header ---------- */
.dash-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 2px 18px 2px;
    border-bottom: 1px solid var(--dash-line);
    margin-bottom: 22px;
}
.dash-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.14em;
    color: var(--dash-safe);
    text-transform: uppercase;
}
.dash-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 30px;
    color: var(--dash-text);
    margin: 2px 0 0 0;
    letter-spacing: -0.01em;
}
.dash-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    color: var(--dash-text-dim);
    max-width: 480px;
    text-align: right;
    line-height: 1.45;
}

/* ---------- Camera panels ---------- */
.camera-frame {
    background: var(--dash-panel) !important;
    border: 1px solid var(--dash-line) !important;
    border-radius: 14px !important;
    padding: 10px !important;
    overflow: hidden;
}
.panel-label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 4px 10px 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: var(--dash-text-dim);
    text-transform: uppercase;
}
.rec-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--dash-alert);
    margin-right: 6px;
    box-shadow: 0 0 0 0 rgba(255, 107, 74, 0.6);
    animation: pulse 1.6s infinite;
}
.scan-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--dash-safe);
    margin-right: 6px;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255, 107, 74, 0.55); }
    70%  { box-shadow: 0 0 0 7px rgba(255, 107, 74, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 107, 74, 0); }
}
@media (prefers-reduced-motion: reduce) {
    .rec-dot { animation: none; }
}

.camera-frame img, .camera-frame video {
    border-radius: 9px !important;
    border: 1px solid var(--dash-line) !important;
    background: var(--dash-panel-2) !important;
}

/* ---------- Telemetry strip ---------- */
.telemetry-row {
    display: flex;
    align-items: stretch;
    gap: 10px;
    flex-wrap: wrap;
    padding: 4px 0 2px 0;
}
.chip {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 2px;
    background: var(--dash-panel);
    border: 1px solid var(--dash-line);
    border-left: 3px solid var(--dash-safe);
    border-radius: 10px;
    padding: 8px 16px;
    min-width: 84px;
}
.chip-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--dash-text-dim);
}
.chip-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: var(--dash-text);
}
.status-pill {
    display: flex;
    align-items: center;
    margin-left: auto;
    padding: 0 18px;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
    border: 1px solid var(--dash-line);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 9px;
}
.status-safe {
    background: var(--dash-safe-dim);
    color: var(--dash-safe);
    border-color: var(--dash-safe);
}
.status-safe .status-dot { background: var(--dash-safe); }
.status-alert {
    background: var(--dash-alert-dim);
    color: var(--dash-alert);
    border-color: var(--dash-alert);
    animation: pulse 1.6s infinite;
}
.status-alert .status-dot { background: var(--dash-alert); }
.status-idle {
    background: var(--dash-panel-2);
    color: var(--dash-text-dim);
}
.status-idle .status-dot { background: var(--dash-text-dim); }

/* ---------- Reset button ---------- */
.reset-btn {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.04em !important;
    background: var(--dash-panel) !important;
    border: 1px solid var(--dash-line) !important;
    color: var(--dash-text-dim) !important;
    border-radius: 10px !important;
}
.reset-btn:hover {
    color: var(--dash-text) !important;
    border-color: var(--dash-text-dim) !important;
}

/* ---------- Footer ---------- */
.dash-footer {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: var(--dash-text-dim);
    border-top: 1px solid var(--dash-line);
    margin-top: 26px;
    padding-top: 14px;
}
"""


# ---------------------------------------
# Gradio Interface
# ---------------------------------------

with gr.Blocks(title=PAGE_TITLE, css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:

    gr.HTML(
        """
        <div class="dash-header">
            <div>
                <div class="dash-eyebrow">Smarttraffic &middot; Driver Monitoring</div>
                <div class="dash-title">Drowsiness Detector</div>
            </div>
            <div class="dash-subtitle">
                Watches your eyes and mouth in real time via MediaPipe FaceMesh
                to catch blinks, yawns, and prolonged eye closure before they
                become a hazard. Nothing is recorded or saved to disk.
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=True):

        with gr.Column(elem_classes="camera-frame"):
            gr.HTML('<div class="panel-label-row"><span><span class="rec-dot"></span>Your webcam</span></div>')
            live_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                label=None,
                show_label=False,
                container=False,
            )

        with gr.Column(elem_classes="camera-frame"):
            gr.HTML('<div class="panel-label-row"><span><span class="scan-dot"></span>Live analysis</span></div>')
            live_output = gr.Image(
                label=None,
                show_label=False,
                container=False,
            )

    live_telemetry = gr.HTML(build_telemetry_html(None))

    live_reset_btn = gr.Button(
        "↺  Reset session (clear blink / yawn counts)",
        elem_classes="reset-btn",
        size="sm",
    )

    live_state = gr.State(value=None)

    live_input.stream(
        fn=live_frame_handler,
        inputs=[live_input, live_state],
        outputs=[live_output, live_state, live_telemetry],
    )

    live_reset_btn.click(
        fn=reset_live_session,
        inputs=None,
        outputs=[live_state, live_telemetry],
    )

    gr.HTML('<div class="dash-footer">SmartTrafficAI &copy; 2026</div>')


if __name__ == "__main__":

    demo.launch()