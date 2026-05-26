import json
import sys
from pathlib import Path

notebook_path = Path(__file__).resolve().parent / "Training Dataset" / "NutriAI_Model.ipynb"

if not notebook_path.exists():
    print(f"Error: Notebook not found at {notebook_path}")
    sys.exit(1)

with notebook_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

cell_20_code = [
    "# Bagi data asli menjadi training dan validation set sebelum augmentasi\n",
    "train_paths, val_paths, train_labels, val_labels = train_test_split(\n",
    "    all_image_paths, all_labels, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=all_labels\n",
    ")\n",
    "\n",
    "print(f\"Sebelum Augmentasi:\")\n",
    "print(f\"Jumlah gambar training asli: {len(train_paths)}\")\n",
    "print(f\"Jumlah gambar validation: {len(val_paths)}\")\n",
    "\n",
    "# Augmentasi data training\n",
    "data_augmentation = tf.keras.Sequential([\n",
    "    tf.keras.layers.RandomFlip(\"horizontal\"),\n",
    "    tf.keras.layers.RandomRotation(0.1),\n",
    "    tf.keras.layers.RandomZoom(0.1),\n",
    "])\n",
    "\n",
    "# Hitung jumlah gambar per kelas di data training\n",
    "train_class_counts = {label: 0 for label in label_to_index.values()}\n",
    "for label in train_labels:\n",
    "    train_class_counts[label] += 1\n",
    "\n",
    "# Tentukan batas minimum per kelas khusus untuk training set (80% dari 300 = 240)\n",
    "MIN_PER_CLASS_TRAIN = 240\n",
    "\n",
    "# Buat salinan list paths & labels training untuk ditambahkan hasil augmentasi\n",
    "augmented_train_paths = list(train_paths)\n",
    "augmented_train_labels = list(train_labels)\n",
    "\n",
    "# Loop tiap label untuk melakukan augmentasi hanya pada training set\n",
    "for label in train_class_counts:\n",
    "    count = train_class_counts[label]\n",
    "    if count >= MIN_PER_CLASS_TRAIN:\n",
    "        continue  # skip kalau sudah cukup\n",
    "\n",
    "    need = MIN_PER_CLASS_TRAIN - count\n",
    "    class_name = index_to_label[label]\n",
    "    \n",
    "    # Ambil list gambar training untuk kelas ini\n",
    "    class_train_paths = [p for p, l in zip(train_paths, train_labels) if l == label]\n",
    "\n",
    "    print(f\"Melakukan augmentasi {need} gambar untuk kelas '{class_name}' di training set...\")\n",
    "\n",
    "    for i in range(need):\n",
    "        src_path = random.choice(class_train_paths)\n",
    "        img_raw = tf.io.read_file(src_path)\n",
    "        img = tf.image.decode_image(img_raw, channels=3)\n",
    "        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])\n",
    "        img = tf.cast(img, tf.float32) / 255.0\n",
    "\n",
    "        # Apply augmentasi\n",
    "        aug = data_augmentation(tf.expand_dims(img, 0), training=True)\n",
    "        aug = tf.squeeze(aug, 0).numpy()\n",
    "        aug = (aug * 255).astype(np.uint8)\n",
    "\n",
    "        # Simpan hasil augmentasi ke folder kelas (dengan prefix)\n",
    "        filename = f\"aug_train_{i}_{os.path.basename(src_path)}\"\n",
    "        save_path = os.path.join(DATA_DIR, class_name, filename)\n",
    "        Image.fromarray(aug).save(save_path)\n",
    "        \n",
    "        # Tambahkan ke list training paths dan labels\n",
    "        augmented_train_paths.append(save_path)\n",
    "        augmented_train_labels.append(label)\n",
    "\n",
    "train_paths = augmented_train_paths\n",
    "train_labels = augmented_train_labels\n",
    "print(f\"\\nSetelah Augmentasi:\")\n",
    "print(f\"Total gambar training (asli + augmentasi): {len(train_paths)}\")\n",
    "print(f\"Total gambar validation (murni bebas augmentasi): {len(val_paths)}\")\n"
]

cell_23_code = [
    "# Membaca ulang dataset tidak diperlukan lagi karena variabel train_paths dan train_labels sudah di-update secara dinamis di memori.\n",
    "# Kami hanya mencetak statistik jumlah gambar untuk memastikan semuanya sinkron.\n",
    "print(f\"Total data latih (train): {len(train_paths)}\")\n",
    "print(f\"Total data validasi (val): {len(val_paths)}\")\n"
]

cell_26_code = [
    "# Pembagian dataset train-test split sudah dilakukan di awal (sebelum augmentasi) untuk menghindari kebocoran data (data leakage).\n",
    "# Variabel train_paths, val_paths, train_labels, dan val_labels siap digunakan langsung di sel berikutnya.\n",
    "print(f\"Siap membuat TensorFlow Dataset.\")\n"
]

modified_cells = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if (
            "MIN_PER_CLASS = 300" in source
            or ("MIN_PER_CLASS_TRAIN = 240" in source and "train_test_split" in source)
        ):
            cell['source'] = cell_20_code
            cell['outputs'] = []
            cell['execution_count'] = None
            modified_cells += 1
            print("Updated Cell 20")
        elif (
            "Total gambar setelah augmentasi" in source
            or "Total data latih (train)" in source
        ):
            cell['source'] = cell_23_code
            cell['outputs'] = []
            cell['execution_count'] = None
            modified_cells += 1
            print("Updated Cell 23")
        elif (
            ("Bagi data menjadi training dan validation set" in source and "stratify=all_labels" in source)
            or "Siap membuat TensorFlow Dataset" in source
        ):
            cell['source'] = cell_26_code
            cell['outputs'] = []
            cell['execution_count'] = None
            modified_cells += 1
            print("Updated Cell 26")

if modified_cells == 3:
    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook successfully fixed and saved!")
else:
    print(f"Warning: Expected to modify 3 cells, but modified {modified_cells}. Notebook was NOT saved.")
