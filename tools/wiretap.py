import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
# Connect to the exact port the probe is broadcasting on
sock.connect("tcp://127.0.0.1:5633")
# Subscribe to EVERYTHING
sock.setsockopt(zmq.SUBSCRIBE, b"")

print("[*] Wiretap active on Port 5633. Waiting for packets...")
while True:
    topic, payload = sock.recv_multipart()
    print(f"\n[RECEIVED] Topic: {topic.decode()}")
    print(f"[PAYLOAD]  {payload.decode()}")
