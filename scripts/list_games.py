"""List recorded games.

Usage:
    python scripts/list_games.py
    python scripts/list_games.py --last 5
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="List recorded games")
    parser.add_argument("--last", type=int, default=0, help="Show only the last N games")
    parser.add_argument("--copy", type=int, default=0, help="Copy game N to clipboard (1=most recent)")
    args = parser.parse_args()

    games_dir = Path(__file__).parent.parent / "games"
    if not games_dir.exists():
        print("No games recorded yet.")
        return

    games = sorted(games_dir.glob("*.pgn"), reverse=True)
    if not games:
        print("No games recorded yet.")
        return

    if args.copy > 0:
        idx = args.copy - 1
        if idx >= len(games):
            print(f"Only {len(games)} games available.")
            return
        pgn = games[idx].read_text()
        import subprocess
        subprocess.run(["pbcopy"], input=pgn.encode(), check=True)
        print(f"Copied to clipboard: {games[idx].name}")
        print(pgn)
        return

    if args.last > 0:
        games = games[:args.last]

    print(f"{'#':<4} {'Date':<20} {'Players':<30} {'Moves':<8} {'File'}")
    print("-" * 80)
    for i, game_path in enumerate(games):
        name = game_path.stem
        # Parse move count from file
        content = game_path.read_text()
        move_count = content.count(".")
        print(f"{i+1:<4} {name:<50} ~{move_count:<6} {game_path.name}")


if __name__ == "__main__":
    main()
