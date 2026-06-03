#!/usr/bin/env python3
"""
run_all_markets.py

Master runner: fetch data, run WFV, compare, and visualize for all 4 markets.

Usage:
    python run_all_markets.py --folds 6
    python run_all_markets.py --folds 6 --skip-data  # If data already cached
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

MARKETS = ['india', 'us', 'brazil', 'uk']

def run_command(cmd: list, description: str):
    """Run a subprocess command with live output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            text=True
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ Failed after {elapsed:.1f}s: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run full multi-market pipeline")
    parser.add_argument(
        '--folds',
        type=int,
        default=6,
        help='Number of WFV folds (default: 6)'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data fetch if already cached'
    )
    parser.add_argument(
        '--markets',
        nargs='+',
        default=MARKETS,
        choices=MARKETS,
        help='Markets to run (default: all 4)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MULTI-MARKET WFV PIPELINE")
    print("=" * 70)
    print(f"Markets: {', '.join(args.markets)}")
    print(f"Folds: {args.folds}")
    print(f"Skip data fetch: {args.skip_data}")
    print("=" * 70)
    
    python = sys.executable
    
    # Phase 1: Data fetch
    if not args.skip_data:
        success = run_command(
            [python, "fetch_market_data.py", "--markets"] + args.markets,
            "PHASE 1: FETCH MARKET DATA"
        )
        if not success:
            print("\n❌ Data fetch failed. Aborting.")
            return
    else:
        print("\n⏭️  Skipping data fetch (--skip-data)\n")
    
    # Phase 2: Run WFV for each market
    wfv_results = {}
    
    for i, market in enumerate(args.markets, 1):
        success = run_command(
            [python, "run_multi_market_wfv.py", "--market", market, "--n_folds", str(args.folds)],
            f"PHASE 2.{i}: RUN WFV — {market.upper()}"
        )
        wfv_results[market] = success
        
        if not success:
            print(f"\n⚠️  {market.upper()} WFV failed, continuing with other markets...")
    
    # Phase 3: Compare results
    successful_markets = [m for m, success in wfv_results.items() if success]
    
    if not successful_markets:
        print("\n❌ No successful WFV runs. Cannot compare results.")
        return
    
    success = run_command(
        [python, "compare_markets.py", "--markets"] + successful_markets,
        "PHASE 3: COMPARE MARKETS"
    )
    
    if not success:
        print("\n⚠️  Comparison failed, but WFV results are saved.")
    
    # Phase 4: Visualize
    success = run_command(
        [python, "visualize_cross_market.py", "--markets"] + successful_markets,
        "PHASE 4: VISUALIZE"
    )
    
    if not success:
        print("\n⚠️  Visualization failed, but tables are saved.")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    for market, success in wfv_results.items():
        status = "✅" if success else "❌"
        print(f"{status} {market.upper():8s}")
    
    results_dir = Path(__file__).parent.parent / "results"
    print(f"\n📊 Results saved to: {results_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
