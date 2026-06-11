# README snippets for audio-anomaly-detection-structural-testing

Copy the `docs/assets` folder into the repository, then paste whichever sections you want into `README.md`.

## Method overview

<p align="center">
  <img src="docs/assets/readme_method_pipeline.png" alt="Audio anomaly detection workflow: WST features, CAE feature maps, NCC scores and classifier output" width="900">
</p>

<p align="center">
  <em>Workflow used by the repository: WST feature arrays are passed through a CAE, hidden-layer feature maps are compared with normal reference maps using NCC, and lightweight classifiers return a normal/anomalous prediction.</em>
</p>

## NCC feature-map comparison

<p align="center">
  <img src="docs/assets/readme_ncc_concept.png" alt="Conceptual NCC feature-map comparison for normal and anomalous samples" width="900">
</p>

<p align="center">
  <em>Conceptual visual only: the real pipeline computes NCC scores between averaged normal CAE feature maps and incoming sample feature maps.</em>
</p>

## Reported paper performance

<p align="center">
  <img src="docs/assets/readme_reported_metrics.png" alt="Paper-reported accuracy and recall for WST CAE NCC audio anomaly detection" width="720">
</p>

<p align="center">
  <em>High-level performance summary reported in the companion paper. Reproduce local metrics with the quick reproduction workflow before reporting new results.</em>
</p>

## Classifier comparison

<p align="center">
  <img src="docs/assets/readme_classifier_comparison.png" alt="Classifier comparison on two-dimensional NCC features" width="850">
</p>

<p align="center">
  <em>Published two-dimensional NCC classifier comparison. Accuracy alone is not enough because the anomalous class is much smaller than the normal class.</em>
</p>

## Quick reproduction path

<p align="center">
  <img src="docs/assets/readme_quick_reproduction.png" alt="Quick reproduction workflow and generated output files" width="900">
</p>

<p align="center">
  <em>Reviewer-friendly path using public WST arrays and the included pretrained CAE model.</em>
</p>
