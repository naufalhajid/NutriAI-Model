# =========================
# GLOBAL CONSTANTS
# =========================

TARGET_KAL_HARIAN = 2000  # Default average daily calorie needs
MODEL_INPUT_SIZE = (320, 320)  # Must match model training input
MAX_CHAT_QUESTIONS = 5  # Max questions per session

# =========================
# FOOD CLASS NAMES
# =========================
CLASS_NAMES = [
    "Ayam Geprek (1 potong) = 394 kkal (62% lemak, 12% karb, 27% prot)",
    "Ayam Pop (1 potong) = 170 kkal (46% lemak, 11% karb, 43% prot)",
    "Ayam goreng (1 potong) = 391 kkal (50% lemak, 16% karb, 34% prot)",
    "Bakso (1 porsi) = 218 kkal (60% lemak, 15% karb, 25% prot)",
    "Batagor (1 porsi) = 400 kkal (43% lemak, 40% karb, 17% prot)",
    "Bika Ambon (1 potong) = 185 kkal (15% lemak, 80% karb, 5% prot)",
    "Cendol (1 porsi) = 465 kkal (36% lemak, 59% karb, 5% prot)",
    "Dadar Gulung (1 potong) = 139 kkal (43% lemak, 49% karb, 8% prot)",
    "Dendeng (1 porsi) = 123 kkal (57% lemak, 11% karb, 33% prot)",
    "Gorengan (1 potong) = 137 kkal (75% lemak, 19% karb, 6% prot)",
    "Gulai Ikan (1 porsi) = 241 kkal (42% lemak, 8% karb, 50% prot)",
    "Gulai Tambusu (1 porsi) = 204 kkal (39% lemak, 11% karb, 50% prot)",
    "Gulai Tunjang (1 porsi) = 243 kkal (42% lemak, 12% karb, 46% prot)",
    "Ikan Goreng (1 potong) = 192 kkal (23% lemak, 0% karb, 77% prot)",
    "Ketoprak (1 porsi) = 402 kkal (34% lemak, 50% karb, 16% prot)",
    "Klepon (1 buah) = 110 kkal (23% lemak, 72% karb, 5% prot)",
    "Kue Cubit (1 buah) = 70 kkal (14% lemak, 74% karb, 11% prot)",
    "Martabak Manis (1 potong) = 270 kkal (36% lemak, 54% karb, 10% prot)",
    "Martabak Telur (1 porsi) = 203 kkal (38% lemak, 41% karb, 22% prot)",
    "Mie Ayam (1 porsi) = 421 kkal (40% lemak, 44% karb, 16% prot)",
    "Nasi Goreng (1 porsi) = 250 kkal (34% lemak, 51% karb, 15% prot)",
    "Nasi Putih (1 porsi) = 135 kkal (2% lemak, 89% karb, 9% prot)",
    "Onde Onde (1 buah) = 101 kkal (16% lemak, 78% karb, 6% prot)",
    "Pempek (1 porsi) = 234 kkal (24% lemak, 49% karb, 26% prot)",
    "Pepes Ikan (1 potong) = 142 kkal (52% lemak, 0% karb, 48% prot)",
    "Pisang Ijo (1 porsi) = 188 kkal (37% lemak, 59% karb, 4% prot)",
    "Putu Ayu (1 buah) = 23 kkal (31% lemak, 57% karb, 12% prot)",
    "Rendang (1 porsi) = 468 kkal (51% lemak, 9% karb, 40% prot)",
    "Roti Bakar (1 potong) = 138 kkal (16% lemak, 68% karb, 16% prot)",
    "Sate Ayam (1 tusuk) = 34 kkal (58% lemak, 8% karb, 34% prot)",
    "Soto Ayam (1 porsi) = 312 kkal (44% lemak, 25% karb, 31% prot)",
    "Sup Ayam (1 porsi) = 75 kkal (29% lemak, 49% karb, 21% prot)",
    "Telur Balado (1 butir) = 71 kkal (73% lemak, 7% karb, 20% prot)",
    "Telur Dadar (1 porsi) = 93 kkal (71% lemak, 2% karb, 28% prot)",
    "Tempe Bacem (1 potong) = 49 kkal (54% lemak, 17% karb, 29% prot)"
]
