import os

print("=== probe: locate test_passages.jsonl under /kaggle/input ===", flush=True)
found = []
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root[len("/kaggle/input"):].count(os.sep)
    if depth > 8:
        dirs[:] = []
        continue
    if "test_passages.jsonl" in files:
        p = os.path.join(root, "test_passages.jsonl")
        found.append(p)
        print("FOUND " + p, flush=True)
print("=== top-level of /kaggle/input ===", flush=True)
for name in sorted(os.listdir("/kaggle/input")):
    p = os.path.join("/kaggle/input", name)
    print(("DIR " if os.path.isdir(p) else "FILE ") + name, flush=True)
print("=== targeted trace-test checks ===", flush=True)
for p in [
    "/kaggle/input/trace-test-passages",
    "/kaggle/input/datasets/idalextan/trace-test-passages",
    "/kaggle/input/idalextan/trace-test-passages",
    "/kaggle/input/datasets/trace-test-passages",
]:
    print(("OK " if os.path.exists(p) else "MISS") + " " + p, flush=True)
print("found count:", len(found), flush=True)
print("=== end ===", flush=True)
