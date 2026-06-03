# Facial-Recognition
# 🔐 Facial Recognition-Based Entry System for Student Residence Halls

<p align="center">
  <img src="https://img.shields.io/badge/Processing_Speed-<1s%2Fframe-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Face_Detector-MTCNN-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Recognizer-FaceNet_128D-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Published-AJRCOS_2023-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://doi.org/10.9734/AJRCOS/2023/v16i4396">
    📄 Read the Paper (Asian Journal of Research in Computer Science)
  </a>
  &nbsp;|&nbsp;
  <a href="https://doi.org/10.9734/AJRCOS/2023/v16i4396">
    🔗 DOI: 10.9734/AJRCOS/2023/v16i4396
  </a>
</p>

> **Official implementation** of the paper:  
> **"Facial Recognition-Based Entry System for Student Residence Halls: Enhancing Security and Accessibility"**  
> *Published in **Asian Journal of Research in Computer Science**, Vol. 16, Issue 4, pp. 344–353, 2023*

---

## 👥 Authors

**Md. Sabbir Ejaz\*, Sourav Debnath, Mohammad Kamrul Hasan, Md. Mahbubul Alam**

Department of Information and Communication Engineering,  
Noakhali Science and Technology University (NSTU), Noakhali, Bangladesh

\* Corresponding author: `sabbirejaz.ice@nstu.edu.bd`

---

## 📌 Overview

Unauthorized access to student residence halls poses serious risks — including theft, trespassing, and safety threats. Conventional access control systems (key cards, fingerprints, iris scanners) require physical contact and are susceptible to spoofing. This work presents a **real-time, deep learning-based facial recognition entry system** for student residence halls that:

- Automatically **detects and identifies faces** from a live webcam or CCTV video stream
- Grants entry to **registered students** and raises an **alert for unknown individuals**
- Logs **name, arrival time, and date** of every recognized and unrecognized person into a CSV file
- Operates at **less than 1 second per frame** — fully real-time
- Handles variation in **illumination and face orientation**

The pipeline uses **MTCNN** for face detection and **Google FaceNet** (128-D embeddings + Triplet Loss) for face recognition.

---

## 🏗️ System Architecture

```
 ┌──────────────────────────────────────────────────────┐
 │                    TRAINING PHASE                    │
 │                                                      │
 │  Dataset Building (30 students × 10 photos = 300)   │
 │            │                                         │
 │  Image Augmentation → 3000 images                   │
 │  (mirror flip, random crop, ±45° rotation)          │
 │            │                                         │
 │  Face Detection via MTCNN                           │
 │            │                                         │
 │  FaceNet → 128-D Face Embeddings (.pkl file)        │
 └──────────────────────────────────────────────────────┘
                           │
                    Stored Embeddings
                           │
 ┌──────────────────────────────────────────────────────┐
 │                  INFERENCE PHASE                     │
 │                                                      │
 │  Webcam / CCTV → Real-Time Video Frames             │
 │            │                                         │
 │  MTCNN Face Detection                               │
 │  (P-Net → R-Net → O-Net pipeline)                  │
 │            │                                         │
 │  FaceNet → 128-D Embedding of detected face        │
 │            │                                         │
 │  Vector Distance Comparison with stored embeddings  │
 │            │                                         │
 │    ┌───────┴────────┐                               │
 │    │ MATCH (known)  │  NO MATCH (unknown)           │
 │    │                │                               │
 │  Log to CSV file  Trigger ALERT +                   │
 │  (name, time, date) Save photo (time, date)         │
 └──────────────────────────────────────────────────────┘
```

---

## 🧠 Core Models

### 1. MTCNN — Face Detection
Multi-Task Cascaded Convolutional Networks processes frames through a 3-stage cascade:

| Stage | Network | Function                                                    |
|-------|---------|-------------------------------------------------------------|
| 1     | P-Net   | Image pyramid → candidate face windows + bounding box regression |
| 2     | R-Net   | Refines P-Net candidates, eliminates false positives via NMS |
| 3     | O-Net   | Final bounding box + facial landmark (key point) localization |

### 2. FaceNet — Face Recognition
The FaceNet model (Schroff et al., CVPR 2015) is structured in 5 sections:

| Section | Component              | Role                                               |
|---------|------------------------|----------------------------------------------------|
| 1       | Batch Input Layer      | Input face image batch                             |
| 2       | Deep CNN Architecture  | Hierarchical facial feature extraction             |
| 3       | L2-Norm                | Feature normalization                              |
| 4       | Embedding Layer        | 128-dimensional face feature vector               |
| 5       | Triplet Loss Function  | Minimizes within-class distance, maximizes between-class distance |

Each face is represented as a **compact 128-byte embedding** — no retraining required when new students are added.

---

## 📂 Dataset

| Property                   | Value                                                      |
|----------------------------|------------------------------------------------------------|
| Total students             | 30                                                         |
| Raw photos per student     | 10                                                         |
| Raw total photos           | 300                                                        |
| After augmentation         | **3,000 photos** (100 per student)                         |
| Augmentation techniques    | Mirror flipping, random fixed-ratio cropping, ±45° rotation |
| Input source (inference)   | Webcam or CCTV camera                                      |

> The dataset was collected in-house from 30 individuals at NSTU; it is not publicly available.

---

## 📊 System Performance

| Metric                          | Value                              |
|---------------------------------|------------------------------------|
| Processing speed                | **< 1 second per frame**           |
| Real-time capability            | ✅ Yes                             |
| Known individual identification | ✅ Displays name + confidence score|
| Unknown individual handling     | ⚠️ Displays "Unknown" + saves photo|
| Illumination robustness         | ✅ Handles varying lighting        |
| Face orientation robustness     | ✅ Handles varying orientations    |
| Entry log format                | CSV (Name, Time, Date)             |

---

## 🛠️ Requirements

```bash
Python >= 3.7
tensorflow >= 2.x       # FaceNet model backbone
keras
mtcnn                   # MTCNN face detector
opencv-python           # Video capture and frame processing
numpy
scipy                   # Euclidean/cosine distance for embedding comparison
Pillow
pandas                  # CSV logging
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
Facial-Recognition-Entry-System/
│
├── dataset/
│   ├── raw/                            # 300 raw photos (30 students × 10 each)
│   └── augmented/                      # 3000 augmented photos (100 per student)
│
├── models/
│   ├── facenet_model/
│   │   └── 20180402-114759/            # Pre-trained FaceNet weights
│   └── embeddings/
│       └── face_embeddings.pkl         # Stored 128-D embeddings of registered students
│
├── src/
│   ├── build_dataset.py                # Capture and organize student photos
│   ├── augmentation.py                 # Mirror flip, crop, rotation augmentation
│   ├── detect_face.py                  # MTCNN face detection (P-Net → R-Net → O-Net)
│   ├── generate_embeddings.py          # FaceNet: encode all student faces → .pkl
│   ├── recognize.py                    # Real-time recognition: compare embeddings
│   └── logger.py                       # CSV logging (name, time, date) + unknown alert
│
├── entry_logs/
│   ├── known_entries.csv               # Log of recognized individuals
│   └── unknown_alerts/                 # Photos of unrecognized faces (timestamped)
│
├── notebooks/
│   └── demo.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Facial-Recognition-Entry-System.git
cd Facial-Recognition-Entry-System
```

### 2. Build the Student Dataset

Capture 10 photos per student (press `s` to save, `q` to quit):

```bash
python src/build_dataset.py --student_name "StudentName" --num_photos 10
```

### 3. Apply Image Augmentation

Expand from 300 → 3000 photos (100 per student):

```bash
python src/augmentation.py --input dataset/raw/ --output dataset/augmented/
```

### 4. Generate Face Embeddings

Run MTCNN detection + FaceNet encoding on all augmented images:

```bash
python src/generate_embeddings.py \
  --dataset dataset/augmented/ \
  --output models/embeddings/face_embeddings.pkl
```

### 5. Run Real-Time Entry System

```bash
# From webcam
python src/recognize.py --source 0

# From CCTV / IP camera stream
python src/recognize.py --source "rtsp://<camera_ip>/stream"
```

The system will:
- Display identified students' **names and confidence scores** in green boxes
- Display **"Unknown"** in a red box for unrecognized faces
- Log every event to `entry_logs/known_entries.csv`
- Save photos of unknown individuals to `entry_logs/unknown_alerts/`

---

## 📋 CSV Log Format

Entry logs are automatically saved in the following format (as shown in the paper's Fig. 6):

| Name       | Time     | Date      |
|------------|----------|-----------|
| Sourav     | 13:02:03 | 8-Oct-23  |
| Kawshar    | 11:13:02 | 1-Nov-23  |
| Unknown    | 14:30:11 | 2-Nov-23  |

---

## 🆚 Comparison with Related Work

| Method / Study                          | Approach               | Accuracy     |
|-----------------------------------------|------------------------|--------------|
| Prasanna et al. (OpenCV + HOG + SVM)    | HOG + SVM              | 75%          |
| Wang et al. (LBP + SVM)                 | SVM on LFW             | 86%          |
| Raghuwanshi et al. (LDA)               | LDA on ORL DB          | 83.57%       |
| Raghuwanshi et al. (PCA)               | PCA on ORL DB          | 66.07%       |
| Zhu et al. (EATA + OpenCV)             | EATA                   | 94.5%        |
| **Ours (MTCNN + FaceNet)**             | **Deep Learning**      | **Real-time, < 1s/frame** |

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{ejaz2023facial,
  title     = {Facial Recognition-Based Entry System for Student Residence Halls:
               Enhancing Security and Accessibility},
  author    = {Ejaz, Md. Sabbir and Debnath, Sourav and Hasan, Mohammad Kamrul
               and Alam, Md. Mahbubul},
  journal   = {Asian Journal of Research in Computer Science},
  volume    = {16},
  number    = {4},
  pages     = {344--353},
  year      = {2023},
  doi       = {10.9734/AJRCOS/2023/v16i4396},
  issn      = {2581-8260}
}
```

---

## ⚠️ Limitations

As acknowledged in the paper:
- Dataset is limited to **30 students** — scaling to larger populations requires further validation
- OCR-style text extraction for unknown alerts is not yet 100% reliable in all lighting conditions
- Full **edge deployment** (e.g., Raspberry Pi) has not yet been implemented
- The model may struggle in highly **unpredictable environments** (extreme lighting, severe occlusion)

---

## 🔮 Future Work

- Regular **dataset updates** for all enrolled students
- **Push notifications** to security personnel when an unknown face is detected
- Reduce **CPU usage and model size** for edge device deployment
- Improve robustness to **unconstrained environments** (extreme angles, partial occlusion, low light)
- Investigate error scenarios and model refinement strategies

---

## 📜 License

This work is published as **Open Access** under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
© 2023 Ejaz et al. Published by Science Domain International.  
See: [http://creativecommons.org/licenses/by/4.0](http://creativecommons.org/licenses/by/4.0)

> Peer review history: [https://www.sdiarticle5.com/review-history/110323](https://www.sdiarticle5.com/review-history/110323)

---

## 🙏 Acknowledgements

The authors declare no competing interests. All authors contributed equally to this work.

---

<p align="center">Made with ❤️ for safer student residence halls</p>
