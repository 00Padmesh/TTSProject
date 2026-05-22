# Multimodal Emotion-Conditioned Speech Synthesis Engine

An end-to-end multimodal Deep Learning and Digital Signal Processing (DSP) pipeline that dynamically conditions text-to-speech generation on visual features. The engine isolates spatial facial geometry from input images, maps the visual traits into a high-dimensional acoustic latent space using a custom trained PyTorch projection adapter, and uses a transformer-vocoder network to synthesize continuous audio waveforms whose prosody and characteristics match the visual context.

---

## 🏗️ System Architecture & Pipeline Flow

The execution framework is completely modularized into four continuous stages:

1. **Computer Vision Front-End:** Ingests raw images and runs an OpenCV Haar Cascade face detection pipeline to extract region-of-interest (ROI) boundary coordinates. This strips away clothes, lighting variations, and background textures to protect downstream embeddings from out-of-distribution noise.
2. **Latent Space Projection:** The isolated facial crop is resized ($224 \times 224 \times 3$), normalized, and mapped through a pre-trained `MobileNetV3` backbone. A custom `nn.Linear` projection layer performs a matrix transformation to adapt the core visual feature vectors into a balanced 512-dimensional acoustic embedding space.
3. **Neural Audio Synthesis:** The 512-d visual embedding behaves as a style anchor inside a pre-trained `SpeechT5` acoustic transformer model. Characters from incoming text strings are tokenized into phoneme sequences, combined with the vision vector via cross-attention layers to build an expressive Mel-Spectrogram, and decoded into raw 16kHz `.wav` waveforms using a `HiFi-GAN` neural vocoder.
4. **Statistical DSP Evaluation:** The system loads synthesized audio tracks via `librosa`, executing a YIN auto-correlation time-frequency algorithm to capture the Fundamental Frequency ($F_0$ Contour) frame-by-frame. It filters unvoiced artifacts below a strict 110 Hz threshold to calculate clean, authentic mathematical mean and variance metrics.

---

## 🧮 Mathematical Alignment Strategy

Visual and acoustic spaces inhabit separate mathematical topologies. To align them, the project employs a cross-modal adapter optimization loop. During the training phase, ground-truth audio target vectors ($A$) are extracted using a frozen, high-fidelity speech encoder (`Wav2Vec2`). 

The weights of the custom visual projection matrix ($W$) are optimized by minimizing the Mean Squared Error (MSE) Loss between the predicted visual style vectors and true speech targets across distinct facial features:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} ||W \cdot V_i - A_i||^2$$

By updating the transformation layer via backpropagation, the network learns to warp its output clusters so that distinct facial profiles shift the numerical vectors into specific coordinate domains that the downstream TTS transformer interprets as targeted pitch, cadence, and vocal properties.

---

## 📈 Evaluation & DSP Verification

To mathematically validate pipeline adjustments, the project integrates an interactive signal tracking block. Without rigorous DSP filtering, silent tracking regions, micro-breaths, and unvoiced consonants drop heavily to the 70–90 Hz spectrum, artificially skewing output analytics. 

By enforcing a **110 Hz high-pass threshold filter**, unvoiced valleys are cleanly eliminated, allowing the system to accurately map genuine spoken performance variations.

### Comparative Baseline Metric Logs

| Visual Input Trigger | Mean Fundamental Frequency ($F_0$) | Pitch Standard Deviation (Vocal Variance) |
| :--- | :--- | :--- |
| **Mock Random Vector Base** | 303.23 Hz | 80.23 Hz (Skewed by Unvoiced Drops) |
| **Filtered Expressive Run** | **319.29 Hz** | **55.47 Hz (True Voiced Signal)** |

---

## 🛠️ Tech Stack & Core Libraries

* **Core Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Computer Vision Layer:** OpenCV (`cv2`), Pillow (`PIL`), Torchvision
* **Speech & Transformer Models:** Hugging Face Transformers (`SpeechT5`, `Wav2Vec2`)
* **Digital Signal Processing:** Librosa, SoundFile (`sf`), NumPy, Matplotlib

---

## 🚀 Future Engineering Roadmap: Explicit Variance Conditioning

While the out-of-the-box base model treats the incoming 512-dimensional vector as a static speaker identity template, the next engineering scale-up involves explicit architectural conditioning on the network's **Variance Adaptors**. 

By slicing the projected 512-dimensional visual tensor into distinct structural sub-components, we can explicitly scale the network's micro-predictors for **Pitch ($F_0$)**, **Duration (speech rate)**, and **Energy (amplitude volume)** before the spectrogram passes to the vocoder, forging a direct linear relationship between explicit muscle contractions on a face and physical properties of the sound wave.
