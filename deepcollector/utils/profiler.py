# =============================================================================
# Profiler Utility
# =============================================================================
import time
from collections import defaultdict
from functools import wraps
import pandas as pd

class Profiler:
    """A simple profiling utility to track execution time and counts."""
    def __init__(self):
        self.stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})

    def track(self, category):
        """Decorator to track the execution of a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    self.update_stats(category, duration)
            return wrapper
        return decorator

    def update_stats(self, category, duration, count=1):
        """Updates the statistics for a given category."""
        self.stats[category]['count'] += count
        self.stats[category]['total_time'] += duration

    def get_report(self) -> pd.DataFrame:
        """Generates a pandas DataFrame report of the profiling statistics."""
        if not self.stats:
            return pd.DataFrame(columns=['Category', 'Count', 'Total Time (s)', 'Avg Time (s)'])

        report_data = []
        for category, data in self.stats.items():
            avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
            report_data.append({
                'Category': category,
                'Count': data['count'],
                'Total Time (s)': data['total_time'],
                'Avg Time (s)': avg_time
            })

        df = pd.DataFrame(report_data)
        # Sort by Total Time descending
        df = df.sort_values(by='Total Time (s)', ascending=False).reset_index(drop=True)
        # Format floating point numbers
        df['Total Time (s)'] = df['Total Time (s)'].map('{:.3f}'.format)
        df['Avg Time (s)'] = df['Avg Time (s)'].map('{:.4f}'.format)
        return df

    def reset(self):
        """Resets the profiler statistics."""
        self.stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})

# Create a global instance for easy access
profiler = Profiler()
print("✅ deepcollector/utils/profiler.py written.")