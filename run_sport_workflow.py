"""
Run the full MLB optimization workflow non-interactively.
Usage: python run_sport_workflow.py

v3 workflow with 7-phase per-model mega optimizer:
  Step 1: python main.py (baseline runs automatically)
  Step 2-4: autoopt (grid->genetic->bayesian Elo optimization)
  Step 5: results
  Step 6: backtest (refit Platt with best params)
  Step 7: mega (mega-ensemble baseline backtest)
  Step 8: mega optimize (7-phase: baseline -> per-model solo -> tournament -> meta -> DE -> ablation -> validation)
  Step 9: purgedcv -> pbo -> montecarlo (validation)
  Step 10: kelly (position sizing)
"""
import sys, os, subprocess, time, datetime

def run_workflow():
    sport_path = os.path.dirname(os.path.abspath(__file__))
    sport_name = "MLB"
    log_file = os.path.join(sport_path, f"{sport_name.lower()}_optimization_log.txt")

    lines = []
    lines.append("autoopt")
    lines.append("results")
    lines.append("backtest")
    lines.append("mega")
    lines.append("mega optimize")
    lines.append("purgedcv")
    lines.append("pbo")
    lines.append("montecarlo")
    lines.append("")
    lines.append("kelly")
    lines.append("")
    lines.append("")
    lines.append("quit")

    stdin_data = "\n".join(lines) + "\n"

    print(f"[{datetime.datetime.now():%H:%M:%S}] Starting {sport_name} full optimization workflow...")
    print(f"  Log: {log_file}")
    print()

    start = time.time()

    with open(log_file, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"=== {sport_name} FULL OPTIMIZATION WORKFLOW v3 ===\n")
        log.write(f"Started: {datetime.datetime.now()}\n\n")
        log.flush()

        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py"],
            cwd=sport_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        proc.stdin.write(stdin_data)
        proc.stdin.close()

        for line in proc.stdout:
            log.write(line)
            log.flush()
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in [
                "baseline", "accuracy", "best ", "phase", "optimiz",
                "grid", "genetic", "bayesian", "mega", "ablation",
                "purged", "pbo", "monte carlo", "kelly", "sharpe",
                "log_loss", "log loss", "brier", "deflated",
                "winner", "apply", "auto-optimize", "completed",
                "tune", "improved", "per-model", "tournament",
                "leaderboard", "helps", "hurts", "solo",
                "differential", "coordinate", "validation",
            ]):
                elapsed_m = (time.time() - start) / 60
                print(f"  [{sport_name} {elapsed_m:.0f}m] {line.rstrip()}")

        proc.wait()

        elapsed = time.time() - start
        summary = f"\n=== {sport_name} COMPLETED in {elapsed/60:.1f} minutes (exit code: {proc.returncode}) ===\n"
        log.write(summary)
        print(summary)

    return log_file


if __name__ == "__main__":
    run_workflow()
