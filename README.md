# Bubo 5-Tier Sterile Android Chassis
# aka a "Droid" :)

Foundational hardware and autonomic layer for the Bubo robotics architecture.

Tiers 1-5: Kinematics, Sensor Fusion, Safety, Sim2Real, and the Physical Substrate.

🧠 Architecture Overview: The 5-Tier Sensory Baseline

This repository implements a Sterile Afferent Pathway, demonstrating how high-frequency physical stimuli are transformed into observable cortical perceptions within the Bubo framework.
The Data Flow Pipeline

The system operates across multiple layers of the Bubo abstraction, ensuring total decoupling between the peripheral stimulus and the telemetry output.

    Level 0: Peripheral Stimulus (tools/thalamic_probe.py)

        Role: Emulates a spinal afferent nerve.

        Mechanism: Injects randomized pressure packets (0.70 N to 0.95 N) at 2 Hz directly into the VPL (Ventral Posterior Lateral) relay.

    Level 1: Cortical Processing (bubo/nodes/sensory/s1_node.py)

        Role: The Somatosensory Cortex (S1).

        Mechanism: Polls the relay at 10 Hz, applying cortical persistence to the input. It wraps raw data into a NeuralMessage envelope, injecting real-time neuromodulatory state (DA, 5HT, NE) before broadcasting.

    Level 2: Telemetry Bridge (telemetry/bubo_bridge.py)

        Role: The Translator.

        Mechanism: A ZeroMQ-to-HTTP bridge that scrapes the high-speed cortical broadcast and exposes it as a Prometheus-compatible gauge. It effectively bridges the gap between asynchronous robotics and synchronous observability.

Key Technical Patterns

    NeuralMessage Wrapping: All data is encapsulated with metadata including source, vlan, and neuromod levels.

    Temporal Decoupling: The stimulus (2 Hz) and the cortical broadcast (10 Hz) operate on independent clocks, simulating biological sensory sampling.

    Observability: Integrated with Prometheus/Grafana for real-time "Neural Sparklines."
