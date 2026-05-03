# Crowd Counting with Enhanced MC-CNN

---

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

Proyek ini merupakan pengembangan model *Computer Vision* untuk memprediksi jumlah kerumunan orang (Crowd Counting) dalam sebuah gambar berbasis estimasi peta kepadatan (*density map*). 

Pendekatan *density map* digunakan karena lebih tangguh dalam menangani kasus oklusi (objek saling menutupi) dan variasi skala kepala manusia dalam kerumunan dibandingkan dengan model deteksi objek tradisional (seperti YOLO atau SSD).

---

## Metodologi & Arsitektur Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#fff'}}}%%
graph TB
    %% Style and Shape Classes
    classDef database stroke:#333,stroke-width:2px,fill:#eee,rx:10,ry:10;
    classDef common fill:#fff3e0,stroke:#e67e22,stroke-width:2px,rx:5,ry:5,color:black;
    classDef trainData fill:#ffe0b2,stroke:#d35400,stroke-width:2px,rx:5,ry:5,color:black;
    classDef model fill:#3498db,stroke:#2980b9,stroke-width:2px,rx:5,ry:5,color:white;
    classDef tuning fill:#f39c12,stroke:#d35400,stroke-width:2px,rx:5,ry:5,color:black;
    classDef testData fill:#e1bee7,stroke:#8e44ad,stroke-width:2px,rx:5,ry:5,color:black;
    classDef output fill:#81d4fa,stroke:#4fc3f7,stroke-width:2px,rx:5,ry:5,color:black;

    %% Nodes and Flow
    A[("Private Dataset:<br>Images & JSON Points")]:::database --> B["Parsing Data &<br>Adaptive Sigma Calculation"]:::common
    B --> C["Generate Target<br>Density Maps"]:::common
    C --> D["Data Augmentation<br>(Blur, Brightness, Contrast)"]:::common
    D --> E{{"Split (70:20:10)"}}
    
    %% Training Phase Connections
    E -.-> F["Train & Validation Data"]:::trainData
    F -.-> G["Enhanced MC-CNN<br>(MobileNetV2 Backbone)"]:::model
    
    %% Interacting with Loss and Optimizer
    G <--> H["Custom Loss Function<br>(0.8 MSE + 0.2 MAE)"]:::tuning
    G <--> I["Adam Optimizer &<br>StepLR Scheduler"]:::tuning
    
    %% Moving to final model
    G -.-> J["Optimal Model Weights"]:::model

    %% Testing Phase Connections
    E --> K["Test Data"]:::testData
    K --> J
    
    %% Output Phase
    J --> L["Density Map Prediction<br>& Head Counting"]:::output
    L --> M[/"Test Evaluation:<br>MAE: 53.37 | RMSE: 150.23"/]:::output

    linkStyle default interpolate linear stroke:#ffffff,stroke-width:4px
```

---

## **Rincian Teknis & Hal yang Perlu Diperhatikan**

Pendekatan dalam notebook ini mengimplementasikan teknik pengolahan gambar tingkat lanjut dan penyesuaian arsitektur model:

1. **Adaptive Sigma untuk Density Map:** 
   Pembuatan *ground truth density map* tidak menggunakan radius tetap. Sistem memanfaatkan algoritma `KDTree` untuk menghitung jarak antar-titik (*head points*) guna menghasilkan nilai `sigma` yang adaptif. Hal ini sangat efektif untuk kerumunan yang padat (perspektif jauh) vs renggang (perspektif dekat).
2. **Arsitektur Enhanced MC-CNN:** 
   Model tidak dibangun dari nol, melainkan menggunakan *backbone* `MobileNetV2` yang ringan dan cepat untuk ekstraksi fitur awal. Fitur ini kemudian diteruskan ke 3 kolom konvolusi paralel dengan *receptive field* yang berbeda (ukuran filter 7x7, 5x5, dan 3x3) agar model bisa mengenali kepala manusia dalam berbagai skala.
3. **Custom Loss Function:** 
   Menggunakan penggabungan bobot (lambda = 0.8). `MSELoss` digunakan untuk menghukum kesalahan secara piksel pada *density map*, sedangkan `L1Loss` digunakan untuk menghukum kesalahan pada total prediksi *head count*.
4. **Dynamic Collation:** 
   Memungkinkan *training* gambar dengan resolusi yang berbeda-beda di dalam satu iterasi dengan melakukan *padding* tensor secara otomatis sesuai ukuran maksimum gambar pada setiap *batch*.

---

## **Hasil Evaluasi Model**

### Training & Validation History
Model dilatih selama 100 epoch dengan mekanisme *early stopping* pada patience 15. Dari grafik di bawah, dapat dilihat bahwa *loss* dan *Mean Absolute Error (MAE)* pada *training* dan *validation set* menunjukkan penurunan yang stabil dan konvergen.

![Training and Validation History](mc-cnn-visualize/training_history.png)


### Analisis Kesalahan (Error Analysis)
Grafik sebaran ini membandingkan prediksi jumlah orang (*Predicted Count*) dengan jumlah sebenarnya (*True Count*). Titik-titik biru yang berada di dekat garis merah putus-putus menunjukkan bahwa prediksi model cukup akurat, terutama untuk kerumunan dengan jumlah di bawah 500 orang. Namun, model cenderung melakukan *under-estimation* pada kerumunan yang sangat padat (>1000 orang).

![Error Analysis](mc-cnn-visualize/error_analysis.png)


### Contoh Prediksi Peta Kepadatan (Density Map)
Berikut adalah visualisasi hasil prediksi model pada *Test Set*. Kolom pertama adalah gambar asli, kolom kedua adalah prediksi *density map* dari model, dan kolom ketiga menampilkan *overlay* kepekatan prediksi (Predicted: xx.xx) terhadap jumlah sebenarnya (True: xx.xx). Area dengan warna lebih hangat (merah/oranye) menunjukkan tingkat konsentrasi kerumunan yang tinggi.

![Density Map Predictions](mc-cnn-visualize/predictions.png)


### Metrik Evaluasi Akhir
*   **Validation MAE:** 45.48
*   **Validation RMSE:** 108.74
*   **Test MAE:** 53.37
*   **Test RMSE:** 150.23

---
*Dibuat oleh Muhammad Abil Hasan*
```
