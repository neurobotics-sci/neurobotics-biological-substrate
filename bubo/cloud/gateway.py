# ... (Imports and create_app start)
    @app.route("/api/v1/health")
    def health():
        _name = getattr(profile, "instance_name", "Bubo Research Node")
        return jsonify({
            "status":    "ok",
            "version":   "1.0-sterile",
            "name":      _name,
            "profile":   profile.name,
            "substrate": profile.substrate,
            "llm":       profile.llm_backend,
            "timestamp": time.time(),
        })

    # NOTE: Bond channel endpoints (/api/v1/bond/...) removed for sterile release.

    @app.route("/api/v1/vision", methods=["POST"])
    def vision_endpoint():
        # ... (API key logic)
        system_prompt = "You are an open-source research humanoid robot. Describe what you see accurately and concisely."
        # ... (Payload logic)
# ... (Rest of file remains same)
