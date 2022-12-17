class DataRecorder:
    """
    Record data once per timestep
    """

    def __init__(self):
        self.performance = []

    def record(self, overall_performance):
        self.performance.append(overall_performance)
