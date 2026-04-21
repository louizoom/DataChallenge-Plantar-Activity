# Theoretical Analysis: Connecting Data Science Concepts to Project Architecture

This document highlights the theoretical fundamentals and justifies the architectural and methodological choices made in this plantar activity classification project.

---

## 1. Model Validation: Avoiding Bias
*(Ref: `main_document.pdf` - Arthur Louchart)*

In statistical learning, the necessity of **Cross-Validation** is paramount to *"reduce overfitting and evaluate reliably"*.

- **The initial problem:** For our first model (the Random Forest Baseline), we performed a simple `Train/Test` split (Frame-by-frame). The observed problem (an artificially high score of ~80%) is what is defined as a random split bias. Temporal windows (frames) from the same second end up distributed in both training and testing, which completely skews the result.
- **The structured solution (Group K-Fold):** The solution is applying separation by **Subjects** (*"Subject 1, Subject 2, Subject 3..."*). This is why the `src/evaluation/train_kfold.py` and `src/evaluation/benchmark_kfold.py` scripts were created. By applying the **Group K-Fold** strategy, we ensured that the windows of a single patient (S01, S02...) are **never** fractured between training and evaluation. Our evaluation guarantees that the network is capable of generalizing its learning to a **new, unknown patient**.

---

## 2. Transitioning to Deep Learning
*(Ref: `introduction_course_presentation.pdf` & `course_part_b_presentation.pdf` - Benjamin Allaert)*

Conceptually, standard Machine Learning (ML) reaches its limits on pure time series if it is not assisted with extensive manual feature calculation.

- **The ML glass ceiling:** Dense networks (MLP) and Random Forests had no *"spatio-temporal awareness"*. They evaluate observations independently.
- **Applying Deep Learning:** The course indicates that we must use more complex and deep architectures, equipped with non-linear functions (*"Activation functions like ReLU to make it possible to solve more complex problems"*). This is why we transitioned to **Convolutional Networks (CNN 1D)**. Convolutional Layers analyze ("filter") the multivariate data (the 50 sole sensors) across the temporal dimension via a sliding window (`Window_Size = 50`) to extract dynamic signatures invisible to classic Machine Learning.

---

## 3. The Loss Function
*(Ref: `course_part_b_presentation.pdf` - Benjamin Allaert)*

The course is very clear on this topic (Slide 9): *"We only use the cross-entropy loss function in classification task"*.

- **Optimization choice:** Our challenge is a multi-class categorization task of human behaviors. For our algorithm to "learn" from its mistakes (backpropagation), the optimizer in all scripts uses the canonical module: `nn.CrossEntropyLoss()`.
- **Context adaptation (Balancing):** Given that a "Transition" or "Jump" action is incredibly rarer than the continuously present "Walking" action, we completed the theory by injecting "class weights" (`class_weight='balanced'`) into this CrossEntropy function. This informs the mathematical model that an error on a minority class must be penalized more severely.

---

## 4. The PyTorch Environment
*(Ref: `introduction_to_pytorch.pdf` - José Mennesson)*

The introduction to `Tensors` and PyTorch syntax is the technological backbone.

- **Native Development:** The project completely substituted Pandas DataFrame objects (made for cleaning) in favor of PyTorch's `TensorDataset` and `DataLoader` for training. Code blocks (`nn.Module`, `optim.Adam`, etc.) are at the core of the implementation.
- **Hardware Acceleration:** Our pipelines automatically manage the transfer of Tensors to GPU memory or Mac Apple Silicon (`MPS`), freeing us from heavy CPU constraints mentioned in the lessons.

---

## 🌟 Synthesis: The Champion Model Architecture (ResNet10_1D)

The `course_part_b_presentation.pdf` warns of the complexity of Deep Neural Networks: adding many layers allows for deep asymmetrical understanding but comes with a severe risk: if the error gradient collapses during backpropagation, the first layers learn nothing anymore (the *Vanishing Gradient*).

By using a **Residual design (ResBlocks)**, we put the architectural solution into practice to counter this phenomenon. The "shortcut" (or identity bypass) added within our `ResBlock1D` blocks allows designing a very deep architecture (**10 Layers !**) without suffocating the gradient flow.

**It is this architecture (the `ResNet10_1D`) that dominates the comparison because it uniquely combines:**
1. **Motif Extraction (CNN 1D)** of plantar pressures over time.
2. **An anti-obstruction design (ResNet)** that overcomes the blockage of overly stacked CNNs.
3. **Rigorous, unbiased evaluation** using the *Group K-Fold* protocol dictated by the lessons.
