# Reproducibility report

This report records a fresh reproduction run of the public pre-trained evaluation workflow.

## Run summary

| Item                      | Value                                                    |
| ------------------------- | -------------------------------------------------------- |
| Workflow                  | Pre-trained CAE feature-map/NCC classifier evaluation    |
| Dataset                   | Public processed WST arrays from Zenodo                  |
| Model                     | `models/pretrained_cae_wst_latent24_structural_audio.h5` |
| Output directory          | `outputs/evaluation_pretrained_review`                   |
| Best classifier           | `knn_k5`                                                 |
| Accuracy                  | `0.9957924263674615`                                     |
| Accuracy, percent         | `99.58%`                                                 |
| Normal recall             | `1.0`                                                    |
| Normal recall, percent    | `100.00%`                                                |
| Anomalous recall          | `0.9240506329113924`                                     |
| Anomalous recall, percent | `92.41%`                                                 |
| Confusion matrix          | `[[1347, 0], [6, 73]]`                                   |

## Commands used

```powershell
conda create -n audio-anomaly-review python=3.11 -y
conda activate audio-anomaly-review

cd C:\Test\audio-anomaly-detection-structural-testing

python -m pip install --upgrade pip
python -m pip install -e ".[dev,deep-learning]"

python scripts/download_data.py --output data
python scripts/check_dataset.py --data-dir data --strict

python scripts/evaluate_feature_map_classifier.py `
  --data-dir data `
  --model models/pretrained_cae_wst_latent24_structural_audio.h5 `
  --output-dir outputs/evaluation_pretrained_review
```

## Metric inspection command

```powershell
@'
import json
from pathlib import Path

path = Path("outputs/evaluation_pretrained_review/classifier_metrics.json")
metrics = json.loads(path.read_text())

best = max(
    metrics.items(),
    key=lambda item: (
        item[1]["classification_report"].get("1", {}).get("recall", 0.0),
        item[1]["accuracy"],
        item[0],
    ),
)

name, result = best
print("Best classifier:", name)
print("Accuracy:", result["accuracy"])
print("Confusion matrix:", result["confusion_matrix"])
print("Normal recall:", result["classification_report"]["0"]["recall"])
print("Anomalous recall:", result["classification_report"]["1"]["recall"])
'@ | python
```

## Result

```text
Best classifier: knn_k5
Accuracy: 0.9957924263674615
Confusion matrix: [[1347, 0], [6, 73]]
Normal recall: 1.0
Anomalous recall: 0.9240506329113924
```

## Interpretation

The fresh reproduction run matches the README reference result: k-nearest neighbours with `k=5` is selected as the best classifier, with approximately `99.58%` test accuracy, `100.00%` normal recall, and `92.41%` anomalous recall.

This confirms that the public processed dataset, included pre-trained model, feature-map NCC scoring pipeline, classifier selection logic, and saved evaluation outputs are working together in a clean reviewer-style environment.

## Scope

This report confirms reproducibility of the public pre-trained evaluation path. It does not claim that the model generalises to all structural testing facilities, all microphone placements, all operating regimes, or safety-critical monitoring decisions without further domain validation.
