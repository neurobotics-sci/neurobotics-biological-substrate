# ... (Top of file remains same)
    def __init__(self, bus: NeuralBus):
        self._bus      = bus
        self._phi      = PhiApproximator(n_nodes=8)
        self._candidates: List[WorkspaceEntry] = []
        self._current  = ConsciousMoment()
        self._history: deque = deque(maxlen=100)
        self._node_activations = np.zeros(8)
        self._running  = False
        self._lock     = threading.Lock()

        # Sterile Research Self-Report Logic
        self._self_model = {
            "identity":     "Bubo Open-Source Research Platform.",
            "capabilities": "Distributed neuromorphic reasoning, sensorimotor integration, and cognitive routing.",
            "limitations":  "This instance is configured as a sterile research baseline."
        }
# ... (Rest of file remains same)
