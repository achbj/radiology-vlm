import json
import matplotlib.pyplot as plt
import os

metrics_file = "training_metrics.json"

if not os.path.exists(metrics_file):
    print("Metrics file not found. Please run: modal volume get vlm-model-vol training_metrics.json .")
    exit(1)

with open(metrics_file, 'r') as f:
    logs = json.load(f)
    
steps = []
losses = []

for log in logs:
    if 'loss' in log:
        steps.append(log.get('step', log.get('epoch')))
        losses.append(log['loss'])

if not steps:
    print("No loss data found in logs.")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o', linestyle='-', color='tab:blue', label='Training Loss')
plt.title('Florence-2 LoRA Fine-Tuning on VQA-RAD (A100-40GB)')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig('vlm_loss_curve.png', dpi=300, bbox_inches='tight')
print("Plot saved as vlm_loss_curve.png")
