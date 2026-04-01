"""
bubo/cloud/gateway.py — Bubo Unified V10
REST + WebSocket API gateway. Active on aws_local and aws_api profiles.
Bridges HTTP/WS clients to the internal ZMQ neural bus.
"""
import os, time, json, logging
from bubo.shared.profile import profile
from bubo.bus.neural_bus import T
from bubo.llm.router import get_router

logger = logging.getLogger("Gateway")

try:
    from flask import Flask, request, jsonify
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if not profile.is_aws:
    logger.debug("Gateway: hardware profile — API gateway not started")


def create_app(bus=None):
    """Create and return the Flask application."""
    if not HAS_FLASK:
        raise ImportError("flask and flask-socketio required: pip install flask flask-socketio")

    app    = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("BUBO_SECRET", "bubo-change-me")
    sio    = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
    router = get_router()

    @app.route("/api/v1/health")
    def health():
        return jsonify({
            "status":    "ok",
            "version":   "9000",
            "profile":   profile.name,
            "substrate": profile.substrate,
            "llm":       profile.llm_backend,
            "nodes":     profile.node_count,
            "timestamp": time.time(),
        })

    @app.route("/api/v1/chat", methods=["POST"])
    def chat():
        data     = request.get_json() or {}
        text     = data.get("message", "").strip()
        affect   = data.get("affect")
        if not text:
            return jsonify({"error": "message required"}), 400
        result = router.query(text, affect=affect, timeout=15.0)
        return jsonify(result)

    @app.route("/api/v1/profile")
    def get_profile():
        return jsonify({
            "name":        profile.name,
            "description": profile.description,
            "substrate":   profile.substrate,
            "llm_backend": profile.llm_backend,
            "node_count":  profile.node_count,
            "has_llm_node":profile.has_llm_node,
            "uses_api_llm":profile.uses_api_llm,
        })

    @app.route("/api/v1/llm/stats")
    def llm_stats():
        return jsonify(router.stats)

    @app.route("/api/v1/llm/backend", methods=["POST"])
    def set_backend():
        data    = request.get_json() or {}
        backend = data.get("backend","")
        try:
            router.set_backend(backend)
            return jsonify({"ok": True, "backend": backend})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # ── WebSocket ────────────────────────────────────────────────────────────
    @sio.on("connect")
    def on_connect():
        emit("bubo_ready", {"message": "Bubo V10 connected.",
                            "profile": profile.name,
                            "llm":     profile.llm_backend})

    @sio.on("speech_input")
    def on_speech(data):
        text   = data.get("transcript", "").strip()
        affect = data.get("affect")
        if text:
            result = router.query(text, affect=affect, timeout=15.0)
            emit("bubo_response", result)

    @sio.on("introspect")
    def on_introspect(data):
        q = data.get("question", "are you conscious?")
        result = router.query(
            q + " Please answer with reference to your IIT Φ value and GWT state.",
            timeout=20.0)
        emit("introspect_response", result)

    return app, sio


def run_gateway():
    """Entry point: start the gateway server."""
    if not profile.is_aws:
        logger.info("Gateway skipped — hardware profile")
        return
    app, sio = create_app()
    port     = int(os.environ.get("GATEWAY_PORT", 443))
    tls_cert = os.environ.get("TLS_CERT", "/etc/bubo/tls/cert.pem")
    tls_key  = os.environ.get("TLS_KEY",  "/etc/bubo/tls/key.pem")
    import pathlib
    ssl_ctx  = (tls_cert, tls_key) if pathlib.Path(tls_cert).exists() else None
    logger.info(f"Gateway starting | port={port} | TLS={'yes' if ssl_ctx else 'no'}")
    sio.run(app, host="0.0.0.0", port=port, ssl_context=ssl_ctx)
