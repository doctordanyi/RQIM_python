class QuadDetector:
    """Abstract class for Quad detector interface definition"""
    def detect_quad(self, img):
        raise NotImplemented("A detector must implement this")
