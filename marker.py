import importlib.util, sys, re, time
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---- config ----
student_file = Path("student_infer.py")
test_dir = Path("C:\\Users\\djati\\OneDrive\\Documents\\AI Project Final\\Test")
# ----------------

# --- controlled import (no imports inside student file) ---
spec = importlib.util.spec_from_file_location("student_infer", student_file)
student_mod = importlib.util.module_from_spec(spec)
sys.modules["student_infer"] = student_mod
spec.loader.exec_module(student_mod)

# inject allowed modules into student namespace
student_mod.torch = torch
student_mod.nn = nn
student_mod.models = models

# --- preprocessing (fixed by you) ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- load model using student's function ---
model, classes = student_mod.build_model(torch, nn, models, None)
model.eval()

# --- evaluation loop ---
pat = re.compile(r"^(\d+)-\d+\.jpe?g$", re.IGNORECASE)
files = sorted(p for p in test_dir.iterdir()
               if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
               and not p.name.startswith(("._", ".")))

total = correct = skipped = 0
sum_infer_s = 0.0
for p in files:
    m = pat.match(p.name)
    if not m:
        skipped += 1
        continue
    true_label = m.group(1)
    try:
        img = Image.open(p).convert("RGB")
        t0 = time.perf_counter()
        idx = student_mod.predict(model, img, preprocess, torch)
        dt = time.perf_counter() - t0
        pred_label = str(classes[idx])
    except Exception as e:
        skipped += 1
        print(f"{p.name:20s} true={true_label:>4s} pred=ERR acc={correct/(total or 1):.4f} [{e}]")
        continue

    total += 1
    correct += int(pred_label == true_label)
    sum_infer_s += dt
    print(f"{p.name:20s} true={true_label:>4s} pred={pred_label:>4s} acc={correct/total:.4f}  [{dt:.3f}s]")

final_acc = correct/total if total else 0.0
print(f"\nEvaluated: {total} Correct: {correct} Skipped: {skipped} Final accuracy: {final_acc:.4f}")
if total > 0:
    avg_s = sum_infer_s / total
    ips = total / sum_infer_s if sum_infer_s > 0 else 0.0
    print(f"Average per-image time: {avg_s:.3f}s | Throughput: {ips:.2f} img/s")