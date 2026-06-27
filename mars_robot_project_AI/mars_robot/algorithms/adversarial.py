# adversarial.py
# Nhom 5: Thuat toan DOI KHANG (Adversarial Search)
# Gom: Minimax va Alpha-Beta Pruning
#
# Bai toan doi khang:
#   - Robot A (MAX): co gang thu NHIEU mau nhat co the
#   - Robot B (MIN): co gang can tra Robot A, thu mau truoc
#   - 2 robot di xen ke nhau (Robot A → Robot B → Robot A → ...)
#   - Ai thu nhieu mau hon thi THANG

import math
from collections import deque
from utils import get_neighbors, manhattan, get_cost, GRID_SIZE, WALL


# =====================================================
# HAM NOI BO
# =====================================================

def _bfs_tim_duong(grid, diem_dau, diem_dich):
    """BFS don gian tim duong ngan nhat"""
    hang_doi = deque()
    hang_doi.append(diem_dau)

    den_tu = {diem_dau: None}

    while hang_doi:
        hien_tai = hang_doi.popleft()

        if hien_tai == diem_dich:
            duong_di = []
            o = hien_tai
            while o is not None:
                duong_di.append(o)
                o = den_tu[o]
            duong_di.reverse()
            return duong_di

        for o_ke in get_neighbors(grid, hien_tai):
            if o_ke not in den_tu:
                den_tu[o_ke] = hien_tai
                hang_doi.append(o_ke)

    return []


def _tim_vi_tri_robot_b(grid, vi_tri_a):
    """
    Tim vi tri xuat phat cho Robot B
    Chon o xa nhat voi Robot A (theo Manhattan)
    De 2 robot bat dau o 2 phia doi dien
    """
    vi_tri_tot_nhat = vi_tri_a
    khoang_lon_nhat = 0

    for hang in range(GRID_SIZE):
        for cot in range(GRID_SIZE):
            if grid[hang][cot] != WALL:
                khoang = manhattan((hang, cot), vi_tri_a)
                if khoang > khoang_lon_nhat:
                    khoang_lon_nhat = khoang
                    vi_tri_tot_nhat = (hang, cot)

    return vi_tri_tot_nhat


def _danh_gia_trang_thai(vi_tri_a, vi_tri_b, cac_mau_con_lai, vi_tri_base,
                          diem_a, diem_b):
    """
    Ham danh gia (heuristic) trang thai hien tai
    Dung khi het do sau hoac het mau

    Score = (diem_a - diem_b) * 100
          + (khoang B den mau gan nhat) - (khoang A den mau gan nhat)
          → Robot A muon score CAO, Robot B muon score THAP
    """
    chenh_lech_diem = diem_a - diem_b

    if not cac_mau_con_lai:
        # Khong con mau → ket qua chi dua vao diem so
        return chenh_lech_diem * 100

    # Uu the vi tri: ai gan mau hon
    khoang_a_den_mau = min(manhattan(vi_tri_a, m) for m in cac_mau_con_lai)
    khoang_b_den_mau = min(manhattan(vi_tri_b, m) for m in cac_mau_con_lai)

    # A muon gan mau hon B → khoang_b - khoang_a duong la tot cho A
    uu_the_vi_tri = (khoang_b_den_mau - khoang_a_den_mau) * 0.5

    return chenh_lech_diem * 100 + uu_the_vi_tri


def _minimax_de_quy(grid, vi_tri_a, vi_tri_b, cac_mau_con_lai,
                    vi_tri_base, diem_a, diem_b,
                    do_sau, alpha, beta, la_max, co_alpha_beta):
    """
    Ham de quy Minimax (co tuy chon Alpha-Beta)

    Tham so:
    - la_max: True neu dang den luot Robot A (MAX), False la Robot B (MIN)
    - co_alpha_beta: True thi su dung cat nhanh alpha-beta
    - alpha: gia tri tot nhat MA MAX co the dam bao
    - beta:  gia tri tot nhat MA MIN co the dam bao

    Cat nhanh Alpha-Beta:
    - beta <= alpha → MIN khong chon nhanh nay (da co phuong an tot hon)
    """
    # Dieu kien dung: het do sau hoac het mau
    if do_sau == 0 or not cac_mau_con_lai:
        return _danh_gia_trang_thai(
            vi_tri_a, vi_tri_b, cac_mau_con_lai,
            vi_tri_base, diem_a, diem_b
        )

    if la_max:
        # Luot Robot A (MAX): chon nuoc di tot nhat (maximize)
        gia_tri_tot_nhat = -math.inf

        for nuoc_di in get_neighbors(grid, vi_tri_a):
            # Kiem tra neu nuoc di nay thu duoc mau
            thu_duoc = 1 if nuoc_di in cac_mau_con_lai else 0
            mau_moi  = cac_mau_con_lai - {nuoc_di}

            gia_tri = _minimax_de_quy(
                grid, nuoc_di, vi_tri_b, mau_moi,
                vi_tri_base, diem_a + thu_duoc, diem_b,
                do_sau - 1, alpha, beta, False, co_alpha_beta
            )

            gia_tri_tot_nhat = max(gia_tri_tot_nhat, gia_tri)

            if co_alpha_beta:
                alpha = max(alpha, gia_tri)
                if beta <= alpha:
                    break  # Cat nhanh! (beta cut-off)

        return gia_tri_tot_nhat

    else:
        # Luot Robot B (MIN): chon nuoc di tot nhat (minimize)
        gia_tri_tot_nhat = math.inf

        for nuoc_di in get_neighbors(grid, vi_tri_b):
            thu_duoc = 1 if nuoc_di in cac_mau_con_lai else 0
            mau_moi  = cac_mau_con_lai - {nuoc_di}

            gia_tri = _minimax_de_quy(
                grid, vi_tri_a, nuoc_di, mau_moi,
                vi_tri_base, diem_a, diem_b + thu_duoc,
                do_sau - 1, alpha, beta, True, co_alpha_beta
            )

            gia_tri_tot_nhat = min(gia_tri_tot_nhat, gia_tri)

            if co_alpha_beta:
                beta = min(beta, gia_tri)
                if beta <= alpha:
                    break  # Cat nhanh! (alpha cut-off)

        return gia_tri_tot_nhat


# =====================================================
# HAM CHAY GAME CHINH
# =====================================================

def _chay_game(grid, start, samples, base, co_alpha_beta, do_sau):
    """
    Chay tro choi doi khang: 2 robot di xen ke theo quyet dinh Minimax

    Moi luot:
    1. Robot A (MAX) chon nuoc di tot nhat bang Minimax
    2. Robot B (MIN) chon nuoc di tot nhat bang Minimax
    3. Lap lai den khi het mau hoac qua nhieu luot
    """
    vi_tri_a = tuple(start)
    vi_tri_b = _tim_vi_tri_robot_b(grid, vi_tri_a)

    # Cac mau con chua ai thu
    cac_mau_con_lai = frozenset(tuple(s) for s in samples)

    diem_a = 0  # so mau Robot A thu duoc
    diem_b = 0  # so mau Robot B thu duoc
    tong_node_duyet = 0

    # Luu lich su di chuyen
    lich_su_a = [vi_tri_a]
    lich_su_b = [vi_tri_b]

    GIOI_HAN_LUOT = 80  # tranh vo han

    print(f"[GAME] Robot A bat dau: {vi_tri_a}")
    print(f"[GAME] Robot B bat dau: {vi_tri_b}")
    print(f"[GAME] So mau: {len(cac_mau_con_lai)}, do sau: {do_sau}")

    # Luu lich su gan day de phat hien lap (6 buoc gan nhat)
    LICH_SU_GAN_DAY = 6
    gan_day_a = []
    gan_day_b = []

    so_luot = 0
    while cac_mau_con_lai and so_luot < GIOI_HAN_LUOT:
        so_luot += 1

        # ========== LUOT ROBOT A (MAX) ==========
        nuoc_di_tot_nhat_a = vi_tri_a
        gia_tri_tot_nhat_a = -math.inf

        for nuoc_di in get_neighbors(grid, vi_tri_a):
            thu_duoc = 1 if nuoc_di in cac_mau_con_lai else 0
            mau_moi  = cac_mau_con_lai - {nuoc_di}

            gia_tri = _minimax_de_quy(
                grid, nuoc_di, vi_tri_b, mau_moi,
                base, diem_a + thu_duoc, diem_b,
                do_sau - 1, -math.inf, math.inf,
                False, co_alpha_beta
            )
            tong_node_duyet += 1

            if gia_tri > gia_tri_tot_nhat_a:
                gia_tri_tot_nhat_a = gia_tri
                nuoc_di_tot_nhat_a = nuoc_di

        # Neu nuoc di tot nhat la vi tri da di qua gan day → dung BFS den mau gan nhat
        if nuoc_di_tot_nhat_a in gan_day_a and cac_mau_con_lai:
            mau_gan_nhat = min(cac_mau_con_lai, key=lambda m: manhattan(vi_tri_a, m))
            duong_bfs = _bfs_tim_duong(grid, vi_tri_a, mau_gan_nhat)
            if len(duong_bfs) >= 2:
                nuoc_di_tot_nhat_a = duong_bfs[1]
                print(f"[A] Luot {so_luot}: Phat hien lap, dung BFS den {mau_gan_nhat}")

        # Cap nhat lich su gan day cua A
        gan_day_a.append(vi_tri_a)
        if len(gan_day_a) > LICH_SU_GAN_DAY:
            gan_day_a.pop(0)

        # Robot A di chuyen
        if nuoc_di_tot_nhat_a in cac_mau_con_lai:
            cac_mau_con_lai = cac_mau_con_lai - {nuoc_di_tot_nhat_a}
            diem_a += 1
            print(f"[A] Luot {so_luot}: Thu mau tai {nuoc_di_tot_nhat_a}! Diem A={diem_a}")

        vi_tri_a = nuoc_di_tot_nhat_a
        lich_su_a.append(vi_tri_a)

        if not cac_mau_con_lai:
            break

        # ========== LUOT ROBOT B (MIN) ==========
        nuoc_di_tot_nhat_b = vi_tri_b
        gia_tri_tot_nhat_b = math.inf

        for nuoc_di in get_neighbors(grid, vi_tri_b):
            thu_duoc = 1 if nuoc_di in cac_mau_con_lai else 0
            mau_moi  = cac_mau_con_lai - {nuoc_di}

            gia_tri = _minimax_de_quy(
                grid, vi_tri_a, nuoc_di, mau_moi,
                base, diem_a, diem_b + thu_duoc,
                do_sau - 1, -math.inf, math.inf,
                True, co_alpha_beta
            )
            tong_node_duyet += 1

            if gia_tri < gia_tri_tot_nhat_b:
                gia_tri_tot_nhat_b = gia_tri
                nuoc_di_tot_nhat_b = nuoc_di

        # Neu nuoc di tot nhat la vi tri da di qua gan day → dung BFS den mau gan nhat
        if nuoc_di_tot_nhat_b in gan_day_b and cac_mau_con_lai:
            mau_gan_nhat = min(cac_mau_con_lai, key=lambda m: manhattan(vi_tri_b, m))
            duong_bfs = _bfs_tim_duong(grid, vi_tri_b, mau_gan_nhat)
            if len(duong_bfs) >= 2:
                nuoc_di_tot_nhat_b = duong_bfs[1]
                print(f"[B] Luot {so_luot}: Phat hien lap, dung BFS den {mau_gan_nhat}")

        # Cap nhat lich su gan day cua B
        gan_day_b.append(vi_tri_b)
        if len(gan_day_b) > LICH_SU_GAN_DAY:
            gan_day_b.pop(0)

        # Robot B di chuyen
        if nuoc_di_tot_nhat_b in cac_mau_con_lai:
            cac_mau_con_lai = cac_mau_con_lai - {nuoc_di_tot_nhat_b}
            diem_b += 1
            print(f"[B] Luot {so_luot}: Thu mau tai {nuoc_di_tot_nhat_b}! Diem B={diem_b}")

        vi_tri_b = nuoc_di_tot_nhat_b
        lich_su_b.append(vi_tri_b)

    # Ca 2 robot cung ve base
    duong_ve_a = _bfs_tim_duong(grid, vi_tri_a, tuple(base))
    duong_ve_b = _bfs_tim_duong(grid, vi_tri_b, tuple(base))

    # Ghep duong ve vao lich su theo tung buoc song song
    max_ve = max(len(duong_ve_a), len(duong_ve_b))
    for i in range(1, max_ve):
        if i < len(duong_ve_a):
            lich_su_a.append(duong_ve_a[i])
        else:
            lich_su_a.append(lich_su_a[-1])

        if i < len(duong_ve_b):
            lich_su_b.append(duong_ve_b[i])
        else:
            lich_su_b.append(lich_su_b[-1])

    # Dam bao 2 lich su bang chieu dai nhau
    while len(lich_su_b) < len(lich_su_a):
        lich_su_b.append(lich_su_b[-1])
    while len(lich_su_a) < len(lich_su_b):
        lich_su_a.append(lich_su_a[-1])

    # Xac dinh ket qua
    if diem_a > diem_b:
        ket_qua = "THANG"
    elif diem_a < diem_b:
        ket_qua = "THUA"
    else:
        ket_qua = "HOA"

    print(f"[GAME] Ket thuc: A={diem_a}, B={diem_b} -> {ket_qua}")
    print(f"[GAME] Tong node da duyet: {tong_node_duyet}")

    return {
        "path":          lich_su_a,
        "robot_b_path":  lich_su_b,
        "robot_b_start": _tim_vi_tri_robot_b(grid, tuple(start)),
        "visited":       [],
        "stats": {
            "steps":          len(lich_su_a) - 1,
            "cost":           sum(get_cost(grid, p) for p in lich_su_a[1:]),
            "visited":        tong_node_duyet,
            "found":          True,
            "score_a":        diem_a,
            "score_b":        diem_b,
            "result":         ket_qua,
            "nodes_explored": tong_node_duyet,
        }
    }


# =====================================================
# HAM CONG KHAI
# =====================================================

def minimax(grid, start, samples, base):
    """
    Minimax khong cat nhanh, do sau = 3
    Duyet nhieu node hon Alpha-Beta nhung don gian hon
    """
    return _chay_game(grid, start, samples, base,
                      co_alpha_beta=False, do_sau=3)


def alpha_beta(grid, start, samples, base):
    """
    Alpha-Beta Pruning, do sau = 4
    Cat nhanh alpha/beta de tinh toan nhanh hon Minimax
    Do sau 4 nhung van nhanh nho cat nhanh hieu qua
    """
    return _chay_game(grid, start, samples, base,
                      co_alpha_beta=True, do_sau=4)
