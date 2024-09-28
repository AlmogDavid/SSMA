import os
import glob
import re


def parse_gps(filename: str) -> float:
    pattern = re.compile(r"^Epoch:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Val:\s*([\d.]+),\s*Test:\s*([\d.]+)$")
    results = []

    with open(filename, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                epoch, loss, val, test = match.groups()
                results.append({
                    "epoch": int(epoch),
                    "loss": float(loss),
                    "val": float(val),
                    "test": float(test)
                })

    min_dict = min(results, key=lambda x: x["test"])
    print(min_dict)

    return min_dict["test"]


if __name__ == "__main__":
    all_gps_files = glob.glob(os.path.join(os.path.dirname(__file__), "gps_*.txt"))
    for gps_file in all_gps_files:
        min_test = parse_gps(gps_file)
        print(f"{gps_file}: {min_test}")