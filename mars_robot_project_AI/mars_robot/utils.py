# utils.py
# Cac ham tien ich dung chung cho toan bo du an Mars Robot
# Bao gom: tao ban do, tinh chi phi, kiem tra hop le, lay o ke

import random
from collections import deque

# =====================================================
# HANG SO (Constants) - cac loai o tren ban do
# =====================================================
EMPTY  = 0   # o trong, co the di qua
WALL   = 1   # tuong, khong the di qua
SAMPLE = 2   # mau vat, robot can thu
BASE   = 3   # tram goc (xuat phat va dich den)
ROUGH  = 4   # dia hinh gon ghe, chi phi cao x5

# Chi phi di vao tung loai o
# NOTE: ROUGH ton 5 don vi thay vi 1 binh thuong
CELL_COST = {
    EMPTY:  1,
    ROUGH:  5,
    SAMPLE: 1,
    BASE:   1,
}

GRID_SIZE   = 15    # ban do hinh vuong 15x15
NUM_SAMPLES = 6     # so luong mau vat tren ban do
WALL_RATIO  = 0.15  # 15% o se la tuong
ROUGH_RATIO = 0.12  # 12% o se la dia hinh gon ghe

# 4 huong di chuyen hop le (len, xuong, trai, phai)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# =====================================================
# CAC HAM TIEN ICH
# =====================================================

def get_cost(grid, vi_tri):
    """
    Tra ve chi phi khi di vao o vi_tri
    ROUGH = 5, cac o khac = 1
    """
    loai_o = grid[vi_tri[0]][vi_tri[1]]
    chi_phi = CELL_COST.get(loai_o, 1)
    return chi_phi


def manhattan(diem_a, diem_b):
    """
    Tinh khoang cach Manhattan giua 2 diem
    Manhattan distance = |row1-row2| + |col1-col2|
    Day la heuristic pho bien trong A* va Greedy
    """
    khoang_hang = abs(diem_a[0] - diem_b[0])
    khoang_cot  = abs(diem_a[1] - diem_b[1])
    return khoang_hang + khoang_cot


def is_valid(grid, hang, cot):
    """
    Kiem tra o (hang, cot) co hop le de di vao khong?
    Dieu kien: nam trong ban do VA khong phai tuong
    """
    # kiem tra bien ban do
    if hang < 0 or hang >= GRID_SIZE:
        return False
    if cot < 0 or cot >= GRID_SIZE:
        return False
    # kiem tra khong phai tuong
    if grid[hang][cot] == WALL:
        return False
    return True


def get_neighbors(grid, vi_tri):
    """
    Tra ve danh sach cac o ke canh hop le cua vi_tri
    Chi xet 4 huong: len/xuong/trai/phai (khong cheo)
    """
    danh_sach_ke = []
    hang_ht = vi_tri[0]
    cot_ht  = vi_tri[1]

    for d_hang, d_cot in DIRECTIONS:
        hang_moi = hang_ht + d_hang
        cot_moi  = cot_ht  + d_cot
        if is_valid(grid, hang_moi, cot_moi):
            danh_sach_ke.append((hang_moi, cot_moi))

    return danh_sach_ke


def is_reachable(grid, diem_dau, diem_dich):
    """
    Kiem tra xem diem_dich co the di den tu diem_dau khong
    Su dung BFS don gian de kiem tra ket noi
    """
    da_tham = set()
    da_tham.add(diem_dau)

    hang_doi = deque()
    hang_doi.append(diem_dau)

    while hang_doi:
        hien_tai = hang_doi.popleft()

        if hien_tai == diem_dich:
            return True  # Tim thay!

        for o_ke in get_neighbors(grid, hien_tai):
            if o_ke not in da_tham:
                da_tham.add(o_ke)
                hang_doi.append(o_ke)

    return False  # Khong co duong di


def path_cost(grid, danh_sach_diem):
    """Uoc tinh tong chi phi di qua cac diem (dung manhattan)"""
    tong = 0
    for i in range(len(danh_sach_diem) - 1):
        tong += manhattan(danh_sach_diem[i], danh_sach_diem[i + 1])
    return tong


# =====================================================
# SINH BAN DO NGAU NHIEN
# =====================================================

def generate_map(seed=None):
    """
    Sinh ban do 15x15 ngau nhien gom:
      - Tuong (WALL): 15% so o
      - Dia hinh gon ghe (ROUGH): 12% so o
      - 6 mau vat (SAMPLE)
      - 1 tram goc (BASE)
      - 1 vi tri robot xuat phat

    Dam bao: robot co the di den tat ca mau vat va base
    Neu khong hop le thi sinh lai (co the lap nhieu lan)

    Returns: (grid, robot_start, samples, base)
    """
    rng = random.Random(seed)
    dem_lan_thu = 0  # debug: xem thu bao nhieu lan moi duoc

    while True:
        dem_lan_thu += 1

        # --- Buoc 1: Tao luoi rong ---
        grid = []
        for i in range(GRID_SIZE):
            hang_moi = []
            for j in range(GRID_SIZE):
                hang_moi.append(EMPTY)
            grid.append(hang_moi)

        # --- Buoc 2: Dat tuong ngau nhien ---
        tong_o    = GRID_SIZE * GRID_SIZE
        so_tuong  = int(tong_o * WALL_RATIO)

        # Lay tat ca vi tri co the dat tuong
        moi_vi_tri = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                moi_vi_tri.append((r, c))

        vi_tri_tuong = rng.sample(moi_vi_tri, so_tuong)
        for r, c in vi_tri_tuong:
            grid[r][c] = WALL

        # --- Buoc 3: Tim cac o con trong ---
        o_trong = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c] == EMPTY:
                    o_trong.append((r, c))

        # Phai co du cho cho robot + base + 6 mau
        if len(o_trong) < NUM_SAMPLES + 2:
            continue  # Qua it o, sinh lai

        # --- Buoc 4: Chon vi tri robot, base, mau vat ---
        vi_tri_chon = rng.sample(o_trong, NUM_SAMPLES + 2)
        robot_start = vi_tri_chon[0]
        vi_tri_base = vi_tri_chon[1]
        danh_sach_mau = vi_tri_chon[2:]  # con lai la mau vat

        # --- Buoc 5: Dat dia hinh gon ghe ---
        o_con_lai = []
        for o in o_trong:
            if o not in vi_tri_chon:
                o_con_lai.append(o)

        so_rough = int(tong_o * ROUGH_RATIO)
        so_rough = min(so_rough, len(o_con_lai))
        vi_tri_rough = rng.sample(o_con_lai, so_rough)
        for r, c in vi_tri_rough:
            grid[r][c] = ROUGH

        # --- Buoc 6: Dat mau vat va base vao luoi ---
        for r, c in danh_sach_mau:
            grid[r][c] = SAMPLE
        grid[vi_tri_base[0]][vi_tri_base[1]] = BASE

        # --- Buoc 7: Kiem tra robot co di den duoc tat ca khong ---
        tat_ca_dich = danh_sach_mau + [vi_tri_base]
        ban_do_ok = True

        for dich in tat_ca_dich:
            if not is_reachable(grid, robot_start, dich):
                ban_do_ok = False
                break

        if ban_do_ok:
            # print(f"[DEBUG] Sinh ban do thanh cong sau {dem_lan_thu} lan thu")
            return grid, robot_start, list(danh_sach_mau), vi_tri_base
        # Chua ok → thu lai vong lap tiep theo
