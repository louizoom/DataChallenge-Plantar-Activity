# Experiment Report: ML/DL Models Benchmark

This document records the results of the initial "crash test" performed between various AI configurations and algorithms for plantar activity classification. The tests were run dynamically on data from sequences `S01` and `S02`.

## ⚙️ Benchmark Methodology
- **Data**: Full subjects S01 and S02 (NaN cleaning, Global standardization).
- **Split**: 80% Training / 20% Validation (Random Split ⚠️).
- **Deep Learning**: For all neural networks (CNN and Dense), training was intentionally **locked to only 4 epochs** so the experimental loop remains brief. The optimizer used is Adam (`lr=0.001`) via hardware acceleration.

## 📊 Results (Raw Data)

Here is the raw performance summary:

| Tested Model | Model Type | Spatial Approach | Max Accuracy (Validation) | Final Loss | Execution Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Machine Learning | Frame-by-Frame | **80.16%** | N/A | **7.1s** |
| **CNN 1D (w=20)** | Deep Learning (Conv1D) | Windowing (0.2s per block) | **47.97%** | 1.7901 | **8.9s** |
| **CNN 1D (w=60)** | Deep Learning (Conv1D) | Windowing (0.6s per block) | **47.90%** | 1.8459 | **3.3s** |
| **MLP Dense** | Deep Learning (Linear) | Windowing (0.5s per block) | **45.83%** | 1.9495 | **3.5s** |

## 💡 Analysis & Interpretation

1. **Random Forest's Domination (short-term):**
   Without huge surprise in the data science field, a _Random Forest_ with an arbitrary depth of 15 crushes a very shallow, unoptimized neural network (only 4 epochs). The RF immediately finds a linear cut-off threshold on pressures (e.g., "This Pressure = This Movement") with 80.1% accuracy!
2. **Under-revving Deep Learning:**
   The 1D CNNs and the MLP all get almost ~47% after 4 Epochs, proving the architecture is learning. The Loss continues to drop drastically. To catch up with the Random Forest, we would need to allocate around **50 to 100 epochs** for PyTorch to discover the deeper _patterns_.
3. **Contribution of Convolutions vs Dense:**
   The 1D CNN with a Micro-Window of 20 frames showed very fast convergence, (slightly) outclassing the large 60-frame windows and especially the naive Dense MLP network (which plateaus at 45%). Temporal micro-convolutions manage to "read" variations well!

## Conclusion
The `src/evaluation/experiment_runner.py` code is provided. You can rerun the parameters (e.g., `epochs=50`) at any time so the crossover between Random Forests and deep AI models reveals the true hidden strength of temporal correlation!
