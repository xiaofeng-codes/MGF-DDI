# <div align="center">MGF-DDI: Knowledge-enhanced multi-granularity feature fusion learning for drug-drug interaction prediction</dib>


### Abstract

<div align="justify">
Combination therapies are widely employed in clinical practice due to their significant efficacy in treating complex medical conditions. However, the rapid increase in the number of available drugs poses a substantial challenge in accurately predicting potential adverse drug-drug interactions (DDIs). This paper proposes MGF-DDI, a knowledge-enhanced multi-granularity feature fusion learning framework. This model constructs a parallel macroscopic-microscopic dual-stream architecture and innovatively incorporates the large-scale pre-trained model Uni-Mol2 for knowledge enhancement. First, in the macroscopic stream, a Graph Transformer with coordinated node-edge updates extracts atomic-level topological features, explicitly integrating 3D molecular geometric features into the attention mechanism to enhance steric conformation perception. Second, in the microscopic stream, chemically meaningful motif graphs are constructed using the BRICS algorithm. To address the physicochemical uncertainty of functional groups in real-world environments, the model integrates a fuzzy graph convolutional network, utilizing Gaussian membership functions to simulate complex chemical mechanisms and capturing more robust microscopic features through dynamic gating. Subsequently, a substructure non-linear interaction module precisely pinpoints high-risk pharmacophores responsible for inducing DDIs. Finally, cross-view contrastive learning aligns macroscopic and microscopic semantics, while a multi-granularity attention mechanism adaptively integrates information from three perspectives, namely the macroscopic global view, the microscopic motif view, and the substructure interaction view, for the final prediction. Extensive experiments demonstrate that MGF-DDI significantly outperforms existing baselines across multiple metrics and exhibits excellent interpretability, providing intuitive, molecular-level insights into the underlying mechanisms of DDIs.</div>


---

### MGF-DDI Framework

![MGF-DDI Framework](framework.png)

---
### Getting Started

#### Prerequisites

To run the code, you'll need the following Python environment setup:

- **Python**: 3.8
- **PyTorch**: 1.12.0+cu113
- **Numpy**: 1.23.0
- **rdkit**: 2024.3.2
- **torch_geometric**: 2.6.1
- **torch_scatter**: 2.1.0+pt112cu113
- **pandas**: 1.3.5

---

### Running the Code

The repository is structured by task. Navigate to the specific task folder to run the model.

1.  **Navigate to the task directory**:
    ```bash
    cd <task_name>
    ```
    (e.g., `cd binary` or `cd multi_rel`)

2.  **Run the training script**:
    ```bash
    python train.py
    ```

For additional arguments and configuration options, please refer to the `train.py` script and the configuration files within each task folder.
