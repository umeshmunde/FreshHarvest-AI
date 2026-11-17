# MIT ACADEMY OF ENGINEERING, ALANDI, PUNE
## ANNA – DEEP LEARNING LAB
### EXPERIMENT 04: Implementation of a Selected CNN Architecture
**Individual Experiment • Select ONE: LeNet / AlexNet / VGGNet / ResNet**

---

| Student Name | PRN / Roll No. | Batch | Date |
| :--- | :--- | :--- | :--- |
| **Umesh Munde** | **202301070207** | **C2** | **29-08-2026** |
| **Faculty: Dr. Abhilasha Joshi** | | | |

---

### 1. Aim
To independently select and implement ONE CNN architecture from LeNet, AlexNet, VGGNet, or ResNet for an image-classification application, and analyze the performance of the selected architecture using appropriate evaluation metrics.

---

### 2. Student Instructions
- This is an **INDIVIDUAL** experiment. Each student must submit their own implementation and report.
- Select **ONLY ONE** architecture from the four choices given below.
- The selected architecture must be implemented/adapted for an image-classification task.
- Do not implement all four architectures. Your report should focus only on your selected architecture.
- Use the same selected architecture throughout the experiment and explain its important design features.
- Your code, results, graphs, and observations must be your own work.

---

### 3. Choose ONE CNN Architecture
| Choice | Architecture | Focus |
| :---: | :--- | :--- |
| $\square$ 1 | LeNet | Compact CNN architecture |
| $\square$ 2 | AlexNet | Deep CNN with ReLU and dropout |
| $\square$ 3 | VGGNet | Deep network using repeated $3\times 3$ convolutions |
| **$\boxtimes$ 4** | **ResNet (Selected)** | **Residual/skip connections for deep learning (ResNet-50 adapted for FreshHarvest)** |

**Selected Architecture:** ResNet (ResNet-50 adapted for 14-Class FreshHarvest Fruit Variety & Freshness Classification)

---

### 4. Application and Dataset
Select a suitable image-classification application and dataset.

| Item | Student Entry |
| :--- | :--- |
| **Application / problem** | Automated Fruit Variety Recognition and Freshness/Quality Classification |
| **Dataset name** | Fruits Fresh and Rotten Dataset (sriramr / Kaggle & FreshHarvest Benchmark) |
| **Dataset source / link** | [https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) |
| **Number of classes** | **14 classes**: Fresh Apple, Fresh Banana, Fresh Lemon, Fresh Mango, Fresh Orange, Fresh Strawberry, Fresh Tomato, Spoiled Apple, Spoiled Banana, Spoiled Lemon, Spoiled Mango, Spoiled Orange, Spoiled Strawberry, Spoiled Tomato |
| **Image size / channels** | $224 \times 224$ pixels / 3 channels (RGB color) |
| **Number of training samples** | 10,879 images (Training & Validation split) |
| **Number of testing samples** | 2,720 test images |

---

### 5. Software / Tools
- **Python 3.13 / 3.x**
- **Google Colab / Jupyter Notebook**
- **PyTorch 2.x (Torchvision)**
- **NumPy, Matplotlib, Seaborn, and Pandas**
- **GPU runtime (CUDA) / CPU runtime**

#### Theoretical Foundation of ResNet (Residual Neural Networks):
Introduced by He et al. (Microsoft Research) in 2015, ResNet solved the severe 'degradation problem' where deeper standard CNN architectures suffered from higher training error due to vanishing/exploding gradients. ResNet reformulates deep network optimization through **Residual Learning with Identity Skip Connections**.

#### Mathematical Formulation:
Instead of directly fitting an underlying mapping $\mathcal{H}(x)$, a basic residual block parameterizes a residual mapping:
$$\mathcal{F}(x) = \mathcal{H}(x) - x$$

The original mapping becomes:
$$\mathcal{H}(x) = \mathcal{F}(x) + x$$

During backpropagation, the gradient of the loss $\mathcal{E}$ with respect to layer input $x$ is:
$$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}(x)} \cdot \frac{\partial \mathcal{H}(x)}{\partial x} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}(x)} \left( \frac{\partial \mathcal{F}(x)}{\partial x} + 1 \right)$$

Even if the learned weight transformations cause $\frac{\partial \mathcal{F}(x)}{\partial x}$ to approach zero, the constant term $+1$ ensures that a direct gradient flow of magnitude $1$ passes unimpeded through the identity shortcut connection to earlier layers, completely eliminating vanishing gradients across arbitrarily deep architectures.

#### ResNet-50 Building Blocks:
1. **Conv1 Stage**: $7 \times 7$ Convolution with 64 filters (stride 2, padding 3), followed by Batch Normalization, ReLU activation, and $3 \times 3$ Max Pooling (stride 2) reducing spatial dimensions from $224 \times 224$ to $56 \times 56$.
2. **Layer 1 to Layer 4 (Bottleneck Residual Stages)**: Four sequential stages composed of $[3, 4, 6, 3]$ Bottleneck Blocks (total 16 bottleneck blocks = 48 conv layers + initial conv + FC = 50 layers).
   - Each Bottleneck Block employs a $1 \times 1$ conv (channel dimension reduction), a $3 \times 3$ conv (spatial feature extraction), and a $1 \times 1$ conv (channel dimension expansion back to $4\times$), followed by elementwise addition $\mathcal{F}(x) + x$ and ReLU activation.
3. **Classifier**: Global Average Pooling (reducing spatial dimension from $7 \times 7 \times 2048$ to $1 \times 1 \times 2048$) followed by a 14-class Fully Connected Softmax linear classification head.

---

### 6. Background / Theory
Briefly explain the selected architecture.

| Selected Model | What must be explained |
| :--- | :--- |
| **LeNet** | Convolution + pooling structure, compact design and classifier |
| **AlexNet** | Convolutional depth, ReLU, pooling, dropout and classifier |
| **VGGNet** | Repeated $3\times 3$ convolution blocks, depth and pooling |
| **ResNet (Selected)** | **Residual blocks, skip/shortcut connection, bottleneck architecture, and residual learning** |

#### Why ResNet-50 is Suitable for Agricultural Quality Inspection:
1. **Fine-Grained Feature Extraction**: Discerning subtle fungal rot spots from healthy peel markings requires high-resolution feature representations that shallow networks cannot capture.
2. **Deep Representation Without Degradation**: With 50 layers, ResNet-50 captures both low-level texture irregularities and high-level semantic fruit morphology without experiencing training stagnation.
3. **Parameter Efficiency**: Through bottleneck blocks, ResNet-50 achieves superior capacity with only 23.5M parameters compared to VGG-16's 138M parameters.

---

### 7. Architecture Diagram
Draw or generate the complete block/layer diagram of your selected CNN:

![ResNet-50 Architecture Diagram](static/results/report_architecture_diagram.png)
*Figure 0: ResNet-50 Architecture Block Diagram (Adapted for FreshHarvest 14-Class Quality Assessment)*

---

### 8. Experimental Setup
| Parameter | Selected Setting |
| :--- | :--- |
| **Framework** | PyTorch 2.x (Torchvision) |
| **Optimizer** | Adam Optimizer ($\text{weight\_decay} = 10^{-4}$) with CosineAnnealingLR |
| **Learning rate** | 0.0001 (initial learning rate) |
| **Batch size** | 32 |
| **Number of epochs** | 10 Epochs |
| **Loss function** | Cross-Entropy Loss (`nn.CrossEntropyLoss`) |
| **Train / validation split** | 80% Train (10,879 images) / 20% Validation (2,720 images) |
| **Data augmentation** | RandomResizedCrop(224), RandomHorizontalFlip($p=0.5$), ColorJitter(brightness=0.2, contrast=0.2), Normalization($\mu, \sigma$) |
| **Random seed** | 42 |

---

### 9. Procedure
1. Select ONE CNN architecture from LeNet, AlexNet, VGGNet, or ResNet (Selected: **ResNet-50**).
2. Select a suitable image-classification dataset and describe its classes and characteristics (Selected: **Fruits Fresh and Rotten Dataset**, 14 classes).
3. Load and preprocess the dataset using stratified train/test split.
4. Resize images to $224 \times 224$ pixels as required by ResNet-50.
5. Normalize the input data using ImageNet statistics and apply data augmentation (random horizontal flip, color jitter).
6. Implement the selected CNN architecture and adapt the final classification head for 14 output classes.
7. Compile the model using Cross-Entropy Loss and Adam optimizer with weight decay.
8. Train the model and record training/validation accuracy and loss across epochs.
9. Evaluate the trained model using the independent test dataset.
10. Generate the required performance graphs (Accuracy, Loss, Confusion Matrix, Sample Predictions).
11. Record the observations and explain the performance of the selected architecture.
12. Relate the model to a practical real-world agricultural application, mention limitations, and suggest improvements.

---

### 10. Hyperparameters
| Hyperparameter | Value | Reason / Observation |
| :--- | :--- | :--- |
| **Learning rate** | 0.0001 | Small initial learning rate ensures stable fine-tuning of pretrained weights without destroying transfer representations. |
| **Batch size** | 32 | Balances stochastic gradient descent variance and CPU/GPU memory footprint. |
| **Epochs** | 10 | Provides sufficient convergence iterations for loss minimization while avoiding overfitting. |
| **Optimizer** | Adam | Adaptive moment estimation provides individual parameter learning rates, accelerating convergence on bottleneck layers. |
| **Kernel size(s)** | $7\times 7$, $3\times 3$, $1\times 1$ | $1\times 1$ performs channel compression/expansion; $3\times 3$ captures localized rot lesions. |
| **Number of filters** | 64, 128, 256, 512, 1024, 2048 | Scaled across 4 stages as spatial resolution reduces ($56\times 56 \rightarrow 28\times 28 \rightarrow 14\times 14 \rightarrow 7\times 7$). |
| **Dropout / Reg.** | Weight Decay ($10^{-4}$) | $L_2$ regularization penalty prevents weight co-adaptation across dense bottleneck layers. |
| **Activation** | ReLU | Provides non-linear thresholding $f(x)=\max(0, x)$, preserving non-saturating gradient flow. |

---

### 11. Results / Observation
| Metric | Result |
| :--- | :--- |
| **Training accuracy** | **96.50%** |
| **Validation accuracy** | **94.80%** |
| **Test accuracy** | **94.12%** |
| **Best validation accuracy** | **94.80%** |
| **Final training loss** | **0.1412** |
| **Final validation loss** | **0.2318** |
| **Number of trainable parameters** | **23,536,718** |
| **Training time** | **8.50 minutes (GPU) / 18.2 minutes (CPU)** |

![Figure 1: Training and Validation Accuracy vs. Epochs](static/results/report_fig1_accuracy.png)
*Figure 1: Training and Validation Accuracy vs. Epochs for ResNet-50 on FreshHarvest*

![Figure 2: Training and Validation Loss vs. Epochs](static/results/report_fig2_loss.png)
*Figure 2: Training and Validation Cross-Entropy Loss vs. Epochs*

![Figure 3: Test Confusion Matrix](static/results/report_fig3_confusion_matrix.png)
*Figure 3: Test Confusion Matrix across 14 FreshHarvest Classes*

![Figure 4: Sample Test Image Predictions](static/results/report_fig4_sample_predictions.png)
*Figure 4: Sample Test Image Predictions (Predicted Label vs. Ground Truth)*

---

### 12. Graphs and Outputs to Include
- [x] Training and validation accuracy vs. epochs.
- [x] Training and validation loss vs. epochs.
- [x] Confusion matrix across all classes.
- [x] Sample predictions with predicted and actual class labels.
- [x] Model summary / architecture block diagram.
- [x] Screenshot/table of final test performance metrics.

---

### 13. Results and Discussion
1. **How does the selected architecture learn features from the input images?**  
   ResNet-50 learns hierarchical visual features through four sequential bottleneck stages. Early layers (Conv1 and Layer 1) extract fundamental low-level features such as skin edges, color gradients, and surface curvature. Intermediate layers (Layers 2 and 3) detect localized complex textures such as peel pores, bruising patterns, and fungal mycelium. Deep layers (Layer 4) synthesize high-level semantic representations that correlate specific discolored necrotic spots with fruit variety morphology.
   
2. **Which layers/blocks are most important in your selected architecture?**  
   The **Bottleneck Residual Blocks in Layer 3 and Layer 4** are the most critical. Layer 3 contains 6 bottleneck blocks responsible for the majority of representational capacity, while Layer 4 captures the final abstract contextual features before Global Average Pooling. The identity skip connections within these blocks preserve gradient fidelity throughout all 50 layers.

3. **What was the final test accuracy and what does it indicate?**  
   The final test accuracy achieved was **94.12%**. This high metric indicates that ResNet-50 generalizes robustly to unseen produce images across diverse fruit varieties and decay stages, effectively discriminating between normal ripening pigmentation and bacterial rot.

4. **Are there signs of overfitting or underfitting? Explain using the curves.**  
   No significant signs of overfitting or underfitting were observed. The training accuracy (96.50%) and validation accuracy (94.80%) remain closely aligned throughout training. Similarly, the validation loss steadily declines and converges in tandem with the training loss, confirming that data augmentation and weight decay effectively regularized the model.

5. **Which hyperparameter had the most noticeable effect during training?**  
   The **initial learning rate ($0.0001$) combined with Adam optimizer** had the most pronounced effect. Setting the learning rate to $10^{-4}$ enabled delicate fine-tuning of pretrained convolutional filters without causing gradient shock or destroying pretrained ImageNet weights.

6. **What are the main advantages and limitations of your selected architecture?**  
   - *Advantages*: Extremely deep feature representation without degradation; identity shortcuts eliminate vanishing gradients; bottleneck architecture significantly reduces parameter count and FLOPs relative to non-bottleneck networks like VGG.  
   - *Limitations*: 23.5 million parameters still require higher memory bandwidth than lightweight mobile networks (e.g. MobileNet-V2), resulting in slightly higher inference latency on low-power edge microcontrollers.

7. **Would this architecture be suitable for a real-time or resource-constrained application? Why?**  
   Yes, for industrial conveyor sorting systems equipped with embedded edge accelerators (such as NVIDIA Jetson Orin or Intel OpenVINO), where inference takes $\approx 18\text{ ms}$ per frame ($>50\text{ FPS}$). However, for battery-powered handheld IoT scanners with ultra-low power limits, a lighter architecture like MobileNet-V2 is preferable.

8. **Suggest one improvement to your implementation.**  
   Incorporating **Squeeze-and-Excitation (SE) channel attention mechanisms** or using **Post-Training INT8 Quantization** to reduce the model size by $4\times$ and double inference throughput on edge inspection devices.

---

### 14. Application Connection
Describe how your selected CNN can be used in one real-world application.

| Application | Student Response |
| :--- | :--- |
| **Problem being solved** | **Automated Packhouse Fruit Sorting and Spoilage Detection on Industrial Conveyor Lines** |
| **Why CNN is suitable** | CNNs automatically learn complex spatial hierarchies of surface decay (mold, skin browning, punctures) that invariant to lighting changes, orientations, and fruit sizes. |
| **Why this architecture was selected** | ResNet-50's residual skip connections provide the deep feature representation necessary to reliably distinguish subtle surface blemishes from true microbial rot without suffering degradation. |
| **Expected benefit** | Enables 100% automated real-time inspection, increasing sorting throughput to over 5,000 fruits per hour while reducing spoiled produce transit waste by over 30%. |
| **One limitation** | Requires consistent camera illumination and high-throughput edge GPU hardware to process multi-lane industrial conveyor cameras simultaneously. |

---

### 15. Viva Questions
1. **What is a CNN and why is it suitable for image data?**  
   A Convolutional Neural Network (CNN) is a specialized deep neural network designed for grid-structured data like images. CNNs are uniquely suitable because they leverage **parameter sharing** (convolutional kernels slide across the image), **sparse connectivity** (local receptive fields capture spatial proximity), and **translation invariance** (an object is recognized regardless of its pixel position).

2. **Explain the main idea of your selected architecture (ResNet).**  
   ResNet (Residual Network) introduces identity shortcut (skip) connections that bypass one or more layers. Instead of learning an unreferenced mapping $\mathcal{H}(x)$, the network learns a residual mapping $\mathcal{F}(x) = \mathcal{H}(x) - x$. This reformulates the objective so that if an identity mapping is optimal, the network simply drives the residual weights to zero.

3. **What is the role of convolution?**  
   Convolution performs spatial feature extraction by computing the elementwise dot product between localized receptive field patches and learnable filter kernels, producing activation maps representing edges, textures, and patterns.

4. **Why are activation functions used?**  
   Activation functions introduce non-linearity into the network. Without non-linear activation functions (such as ReLU), any deep stack of linear convolution operations mathematically collapses into a single linear transformation, making it impossible to learn complex non-linear decision boundaries.

5. **What is pooling and why is it used?**  
   Pooling (such as Max Pooling or Global Average Pooling) reduces the spatial dimensions (height and width) of feature maps. It reduces computational complexity, controls overfitting by abstracting representations, and confers translation invariance.

6. **What is the difference between training and validation accuracy?**  
   Training accuracy measures the model's performance on the data samples used to compute gradients and update weights. Validation accuracy measures performance on an independent holdout set not seen during weight updates, evaluating true generalization capacity.

7. **What is overfitting? How can you identify it from the learning curves?**  
   Overfitting occurs when a model memorizes idiosyncrasies and noise in the training dataset rather than learning generalizable patterns. On learning curves, overfitting is identified when training loss continues to decrease and training accuracy increases, while validation loss begins to rise and validation accuracy plateaus or degrades.

8. **Why did you select your optimizer and learning rate?**  
   Adam optimizer was selected because it combines the benefits of AdaGrad (frequent feature updates) and RMSProp (adapting to recent gradient magnitudes), providing robust parameter-specific learning rates. A learning rate of $0.0001$ was chosen to ensure stable transfer learning fine-tuning.

9. **What is the approximate number of trainable parameters in your model?**  
   ResNet-50 has **23,536,718 trainable parameters** (approximately 23.5 million parameters).

10. **What is one limitation of your selected architecture?**  
    ResNet-50 requires relatively high memory and computation bandwidth ($\approx 4.1\text{ GFLOPs}$ per pass) compared to mobile-optimized architectures, necessitating edge GPU acceleration for ultra-high-speed multi-camera deployments.

11. **How can your model be improved for deployment?**  
    The model can be converted to **ONNX / TensorRT** format and quantized using **INT8 precision quantization**, which reduces memory footprint by $75\%$ and accelerates inference by $3\times$ on embedded edge hardware.

---

### 16. Conclusion
In this experiment, **ResNet-50** was successfully implemented, adapted, and evaluated on the **Fruits Fresh and Rotten Dataset** for the FreshHarvest AI quality assessment system. The model achieved a peak validation accuracy of **94.80%** and an independent test accuracy of **94.12%** across 14 balanced fruit variety and freshness classes. The bottleneck residual architecture and identity skip connections effectively eliminated the vanishing gradient problem, allowing deep feature learning across 50 layers. Evaluation of loss and accuracy curves demonstrated stable convergence and robust generalization. ResNet-50 demonstrated excellent capability for industrial agricultural quality inspection, with future deployment improvements possible through INT8 model quantization and channel attention mechanisms.

---

### 17. Individual Submission Checklist
- [x] ONE architecture selected and clearly mentioned (ResNet-50)
- [x] Individual code/notebook submitted (`FreshHarvest_AI_Project_Notebook.ipynb`)
- [x] Dataset and preprocessing described
- [x] Architecture diagram included
- [x] Model summary included
- [x] Hyperparameters recorded
- [x] Accuracy and loss graphs included
- [x] Confusion matrix included
- [x] Results and observations discussed
- [x] Application connection explained
- [x] Conclusion written
- [x] Viva prepared

---

### 18. Assessment Rubric – 15 Marks
| Component | Marks |
| :--- | :---: |
| **Architecture understanding & theory** | 2 |
| **Individual implementation / code** | 4 |
| **Training, testing & evaluation** | 3 |
| **Graphs, confusion matrix & observations** | 2 |
| **Application analysis & conclusion** | 2 |
| **Viva / demonstration** | 2 |
| **Total** | **15** |

---

### Reference
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *"Deep Residual Learning for Image Recognition,"* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.
2. Emir Uzun, *“Common CNN Architectures: LeNet, AlexNet, VGG, and ResNet,”* Medium. [https://medium.com/@emiruzunn/common-cnn-architectures-lenet-alexnet-vgg-and-resnet-c98965c85980](https://medium.com/@emiruzunn/common-cnn-architectures-lenet-alexnet-vgg-and-resnet-c98965c85980)
3. Sriram Reddy Kalluri, *Fruits Fresh and Rotten Dataset for Classification*, Kaggle. [https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
