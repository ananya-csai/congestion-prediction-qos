"""Step 0 - download the Abilene and GEANT traffic/routing matrices.

Usage:
    python get_data.py

Files are saved in ./data. If they already exist they are not downloaded
again. Total size is about 120 MB.
"""
import sys
import urllib.request

from config import DATA_DIR, DATA_URL

FILES = ["abilene_tm.csv", "abilene_rm.csv", "geant_tm.csv", "geant_rm.csv"]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA_DIR / name
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        url = DATA_URL.format(file=name)
        print(f"[get ] {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # network problems, proxy, etc.
            print(f"Download failed for {name}: {exc}")
            print("Download it manually from https://github.com/Y-debug-sys/DTM "
                  "(folder Data/) and place it in ./data")
            sys.exit(1)
        print(f"[ok  ] {name} ({target.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
