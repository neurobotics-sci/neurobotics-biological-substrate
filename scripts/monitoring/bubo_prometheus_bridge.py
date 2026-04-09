#!/usr/bin/env python3
"""
bubo_prometheus_bridge.py
Translates Bubo's ZMQ NeuralBus telemetry into Prometheus HTTP metrics.
"""

import zmq
import json
import time
import threading
from prometheus_client import start_http_server, Gauge

# Define the Prometheus Metrics we want to track
# Gauges represent values that can go up and down
NODE_TEMP = Gauge('bubo_node_temperature_celsius', 'CPU/GPU Temperature', ['node'])
DOPAMINE_LEVEL = Gauge('bubo_dopamine_level', 'VTA Dopamine Level')
CMAC_UPDATES = Gauge('bubo_cmac_updates_total', 'Cerebellum CMAC Convergence')
VAGUS_HEARTBEAT = Gauge('bubo_vagus_heartbeat_ms', 'Vagus Nerve Loop Latency')

def zmq_listener():
    """Listens to the ZMQ PUB/SUB bus and updates Prometheus metrics."""
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    
    # Connect to your existing telemetry/thermal monitor port (e.g., 5699)
    sub.connect("tcp://localhost:5699")
    
    # Subscribe to specific topics, or "" for everything
    sub.setsockopt_string(zmq.SUBSCRIBE, "SYS_THERMAL")
    sub.setsockopt_string(zmq.SUBSCRIBE, "NM_DA_VTA")
    sub.setsockopt_string(zmq.SUBSCRIBE, "CRB_DELTA")

    print("ZMQ-to-Prometheus Bridge Listening...")

    while True:
        try:
            # Expecting multipart messages: [Topic, JSON Payload]
            topic, msg = sub.recv_multipart()
            payload = json.loads(msg.decode('utf-8'))
            
            topic_str = topic.decode('utf-8')

            # Route the data to the correct Prometheus Gauge
            if topic_str == "SYS_THERMAL":
                node_name = payload.get("target", "unknown")
                peak_c = payload.get("payload", {}).get("peak_C", 0.0)
                NODE_TEMP.labels(node=node_name).set(peak_c)

            elif topic_str == "NM_DA_VTA":
                # Assuming dopamine payloads carry a 'level' float
                da_level = payload.get("level", 0.0)
                DOPAMINE_LEVEL.set(da_level)

            elif topic_str == "CRB_DELTA":
                cmac_updates = payload.get("n_updates", 0)
                CMAC_UPDATES.set(cmac_updates)

        except Exception as e:
            print(f"Bridge parse error: {e}")

if __name__ == '__main__':
    # 1. Start the Prometheus HTTP server on port 8000
    start_http_server(8000)
    print("Prometheus metrics exposed on http://localhost:8000/metrics")
    
    # 2. Start listening to the ZMQ bus in the background
    listener_thread = threading.Thread(target=zmq_listener, daemon=True)
    listener_thread.start()

    # Keep main thread alive
    while True:
        time.sleep(1)
