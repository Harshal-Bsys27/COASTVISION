# COASTVISION Model Evaluation Results

## Model Summary
- Model: YOLOv8 (Custom Trained)
- Layers: 72
- Parameters: 3,006,233
- GFLOPs: 8.1
- Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, PyTorch 2.6.0)

## Dataset
- Images: 1478
- Instances: 2748
- Backgrounds: 0
- Corrupt: 0

## Overall Metrics
- Precision (P): 0.83
- Recall (R): 0.819
- mAP50: 0.865
- mAP50-95: 0.53
- Fitness: 0.5298589351977536

## Per-Class Metrics
| Class              | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|--------------------|--------|-----------|-------|-------|-------|----------|
| Drowning          | 1234  | 1577     | 0.856 | 0.833 | 0.905 | 0.571   |
| Person out of water| 67    | 98       | 0.821 | 0.806 | 0.852 | 0.547   |
| Swimming          | 464   | 1073     | 0.813 | 0.817 | 0.837 | 0.472   |

## Speed Metrics (per image)
- Preprocess: 4.1 ms
- Inference: 12.8 ms
- Loss: 0.0 ms
- Postprocess: 3.1 ms

## Additional Details
- Results Saved To: C:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\runs\detect\val3
- Fitness Score: 0.5298589351977536
- Maps: [0.57094, 0.54668, 0.47196]
- Class Names: {0: 'Drowning', 1: 'Person out of water', 2: 'Swimming'}
- Instances per Class: [1577, 98, 1073]
- Images per Class: [1234, 67, 464]

## Interpretation Notes
- High mAP50 (0.865) indicates good practical accuracy.
- mAP50-95 (0.53) shows room for improvement in precise bounding boxes.
- Best performance on 'Drowning' class, critical for safety.
- Fast inference supports real-time use.

## Artifacts Available
- Confusion Matrix: confusion_matrix.png
- Precision-Recall Curves: precision_recall_curve.png
- Metrics JSON: metrics.json
- Location: runs/detect/val3/

## Quick Answer for Accuracy Questions
"What's the accuracy of your custom model?" respond:  
"The accuracy of our custom YOLOv8 model, measured by mean Average Precision (mAP), is 86.5% at IoU 0.5 (mAP50) and 53% across stricter thresholds (mAP50-95) on the validation dataset of 1478 images. This indicates strong performance for practical drowning detection, with the highest accuracy on the 'Drowning' class at 90.5% mAP50."

This file summarizes the evaluation output for easy reference during presentations or reports.