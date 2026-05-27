# =============================================================================
# V8: Performance Analytics (Workload Aggregation Fix)
# =============================================================================
import time
import pandas as pd
from tabulate import tabulate

class PerformanceAnalyzer:
    """
    Calculates ROI for each phase.
    Tracks:
    - Net Datasets Added
    - Deletions (Deduplication)
    - Cells Filled (Value from [missing] -> Data)
    - Cells Refined (Value changed)
    - Cells Confirmed (Value verified/unchanged)
    """
    def __init__(self):
        self.phases = []
        self.current_phase_start = None
        self.current_items_start = 0
        self.current_phase_name = ""

        self.fill_counts = {}
        self.refine_counts = {}
        self.confirm_counts = {}
        self.deletion_counts = {}

    def start_phase(self, phase_name: str, current_item_count: int):
        self.current_phase_start = time.time()
        self.current_items_start = current_item_count
        self.current_phase_name = phase_name

        self.fill_counts[phase_name] = 0
        self.refine_counts[phase_name] = 0
        self.confirm_counts[phase_name] = 0
        self.deletion_counts[phase_name] = 0

    def record_cell_change(self, change_type: str, count: int = 1):
        if not self.current_phase_name: return

        if change_type == 'FILL':
            self.fill_counts[self.current_phase_name] += count
        elif change_type == 'REFINE':
            self.refine_counts[self.current_phase_name] += count
        elif change_type == 'CONFIRM':
            self.confirm_counts[self.current_phase_name] += count

    def record_deletions(self, count: int):
        if self.current_phase_name:
            self.deletion_counts[self.current_phase_name] += count

    def end_phase(self, current_item_count: int):
        if not self.current_phase_start: return

        duration = time.time() - self.current_phase_start
        items_added = max(0, current_item_count - self.current_items_start)

        filled = self.fill_counts.get(self.current_phase_name, 0)
        refined = self.refine_counts.get(self.current_phase_name, 0)
        confirmed = self.confirm_counts.get(self.current_phase_name, 0)
        deletions = self.deletion_counts.get(self.current_phase_name, 0)

        minutes = duration / 60 if duration > 0 else 0.001

        self.phases.append({
            "Phase": self.current_phase_name,
            "Time (s)": duration,
            "Net Added": items_added,
            "Deletions": deletions,
            "Filled": filled,
            "Refined": refined,
            "Confirmed": confirmed,
            "Total Cell Ops": filled + refined + confirmed,
            "Velocity": items_added / minutes
        })
        self.current_phase_start = None

    def print_report(self):
        if not self.phases: return

        df = pd.DataFrame(self.phases)
        total_time = df["Time (s)"].sum()

        print("\n" + "="*100)
        print("💰 VALUE ANALYSIS REPORT (Tri-State Updates)")
        print("="*100)

        df_display = df.copy()
        df_display["Time %"] = (df_display["Time (s)"] / total_time * 100).map('{:.1f}%'.format)
        df_display["Time (s)"] = df_display["Time (s)"].map('{:.1f}'.format)
        df_display["Velocity"] = df_display["Velocity"].map('{:.2f}'.format)

        cols = ["Phase", "Time (s)", "Time %", "Net Added", "Deletions", "Filled", "Refined", "Confirmed", "Total Cell Ops", "Velocity"]
        print(tabulate(df_display[cols], headers="keys", tablefmt="simple", showindex=False))

        if not df.empty:
            best_phase = df.loc[df["Net Added"].idxmax()]
            if best_phase["Net Added"] > 0:
                print(f"\n💡 Insight: '{best_phase['Phase']}' contributed the most net datasets ({best_phase['Net Added']}).")

            # Evaluate based on Total Operations (Fills + Confirms) rather than just Fills
            most_work_idx = df["Total Cell Ops"].idxmax()
            most_work = df.loc[most_work_idx]

            most_fills_idx = df["Filled"].idxmax()
            most_fills = df.loc[most_fills_idx]

            if most_fills["Filled"] > 0:
                print(f"💡 Insight: '{most_fills['Phase']}' populated the most missing data ({most_fills['Filled']} Fills).")

            if most_work["Total Cell Ops"] > 0 and most_work_idx != most_fills_idx:
                print(f"💡 Insight: '{most_work['Phase']}' handled the heaviest workload, evaluating {most_work['Total Cell Ops']} cells (mostly Confirms).")

print("✅ deepcollector/utils/analytics.py written (V8: Workload Aggregation Fix).")