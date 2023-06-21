import sys
import json

def main(filename):
    with open(filename) as result_file:
        results = json.load(result_file)
    pids = set()
    for result in results["results"].values():
        for pid in result["shot_pids"]:
            pids.add(pid)
    print(pids)
    print(len(pids))

if __name__ == "__main__":
    main(sys.argv[1])
