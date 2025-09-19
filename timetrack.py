import time

class TimingContext:
    def __init__(self, name, timings_dict):
        self.name = name
        self.timings_dict = timings_dict
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.timings_dict[self.name] = duration
