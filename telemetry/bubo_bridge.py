import zmq
import json
from prometheus_client import start_http_server, Gauge

# 1. Define the Prometheus Metrics
S1_PRESSURE = Gauge('bubo_s1_pressure_newtons', 'Real-time pressure felt by S1 Cortex', ['region'])

def run_bridge():
    # 1. Start the Prometheus scrape endpoint
    start_http_server(8000)
    print("[*] Prometheus Bridge Active | Listening on Port 8000")

    # 2. Setup ZMQ
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB) # <--- The missing line!
    sock.connect("tcp://127.0.0.1:5632") 
    sock.setsockopt(zmq.SUBSCRIBE, b"") # Subscribe to everything

    print("[*] Global Sniffing on 5632. Waiting for packets...")

    while True:
        try:
            parts = sock.recv_multipart()
            if len(parts) < 2:
                continue
                
            topic = parts[0]
            payload = parts[1]
            
            # Diagnostic print to see the "On-the-wire" topic
            print(f"[*] RAW RECEIVED | Topic: {topic} | Size: {len(payload)} bytes")
            
            data = json.loads(payload.decode())

            # --- ADD THIS LINE HERE ---
            print(f"DEBUG: Data keys are: {list(data.keys())} | Full data: {data}")
            # --------------------------
            
            # Update Prometheus
            # 1. Reach inside the framework envelope if it exists
            # If 'payload' is a key, that's where our real data lives
            inner = data.get("payload", data)
        
            # 2. Extract the pressure (check both 'pressure_N' and 'pressure')
            pressure_val = inner.get("pressure_N", inner.get("pressure", 0.0))
        
            # 3. Extract the region (check both 'zone_id' and 'region')
            region_val = inner.get("zone_id", inner.get("region", "unknown"))
        
            # 4. Update Prometheus
            S1_PRESSURE.labels(region=region_val).set(float(pressure_val))
            
        except Exception as e:
            print(f"[!] Bridge encountered data error: {e}")

if __name__ == "__main__":
    run_bridge()
