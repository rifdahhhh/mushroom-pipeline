# Submission 1: Mushroom Classification Pipeline  
Nama: Rifdah Hansya Rofifah  
Username dicoding: rifdahhr  

| Kategori | Deskripsi |
|----------|-----------|
| **Dataset** | [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification/) <br>Dataset ini berisi informasi deskriptif tentang 8124 sampel jamur dari 23 kolom kategorikal seperti bentuk tudung, warna, bau, dan habitat. Setiap sampel diberi label apakah jamur tersebut edible (dapat dimakan) atau poisonous (beracun).|
| **Masalah** | Banyak kasus konsumsi jamur liar terjadi karena sulitnya membedakan jamur beracun (*poisonous*) dengan jamur yang aman dikonsumsi (*edible*) hanya melalui pengamatan visual. Ciri-ciri fisik jamur yang mirip membuat klasifikasi manual menjadi tidak akurat dan berisiko tinggi. Oleh karena itu, diperlukan pendekatan berbasis data dan machine learning untuk membantu identifikasi jamur secara otomatis. |
| **Solusi machine learning** | Model klasifikasi biner dikembangkan menggunakan *supervised learning* untuk memprediksi apakah suatu jamur *poisonous* (1) atau *edible* (0). Model ini diimplementasikan secara menyeluruh melalui pipeline **TensorFlow Extended (TFX)** dan dideploy sebagai layanan prediksi menggunakan **TensorFlow Serving**. |
| **Metode Pengolahan** | Proses transformasi dilakukan menggunakan `tensorflow-transform (tft)`:  <br>- Fitur kategorikal → `tft.compute_and_apply_vocabulary`  <br>- Label target → dikonversi menggunakan:```tf.where(tf.equal(inputs[LABEL_KEY], 'p'), 1, 0)``` <br>Dataset dibagi menjadi data pelatihan dan evaluasi. <br>Data kemudian dibungkus dalam fungsi `input_fn` yang menghasilkan `tf.data.Dataset`. |
| **Arsitektur Model** | Model dibangun menggunakan **TensorFlow Keras Sequential API**:  <br>```model = tf.keras.Sequential([    tf.keras.layers.Dense(units_1, activation='relu'),    tf.keras.layers.Dense(units_2, activation='relu'),    tf.keras.layers.Dense(1, activation='sigmoid')])``` <br>**Loss Function:** `binary_crossentropy` <br>**Optimizer:** `adam` (dengan `learning_rate` hasil tuning) <br>**Metrics:** `binary_accuracy` <br>Model diekspor dengan format `SavedModel`
| **Metrik Evaluasi** | **binary_accuracy**: Akurasi model dalam membedakan kelas 0 & 1  <br>**loss**: Indikator kesalahan prediksi |
| **Performa Model** | Model mencapai **akurasi validasi hingga 99.88%**. <br>Contoh hasil prediksi: <br>```response: {"predictions": [[0.998838544]]} Predicted Label: 1 (Poisonous) Probability: 99.88%``` |

