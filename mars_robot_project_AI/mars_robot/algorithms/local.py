# local.py
# Nhom 3: Thuat toan TIM KIEM CUC BO (Local Search)
# Gom: Hill Climbing va Simulated Annealing
#
# Khac cac nhom truoc:
# - Khong tim duong truc tiep A->B
# - Thay vao do: TOI UU THU TU thu mau (6 mau, tim hoàn vi tot nhat)
# - Sau khi co thu tu tot nhat → dung BFS di theo thu tu do

import math
import random
from collections import deque
from utils import get_neighbors, manhattan, get_cost


# =====================================================
# HAM BFS NOI BO (dung de tinh chi phi thuc)
# =====================================================

def _bfs_tinh_chi_phi(grid, diem_dau, diem_dich):
    """
    BFS tinh chi phi THUC TE (dem so buoc, khong phai manhattan)
    Dung de so sanh cac thu tu mau khac nhau
    """
    # Hang doi luu (vi_tri, chi_phi_tich_luy)
    hang_doi = deque()
    hang_doi.append((diem_dau, 0))

    da_tham = set()
    da_tham.add(diem_dau)

    while hang_doi:
        vi_tri, chi_phi = hang_doi.popleft()

        if vi_tri == diem_dich:
            return chi_phi

        for o_ke in get_neighbors(grid, vi_tri):
            if o_ke not in da_tham:
                da_tham.add(o_ke)
                hang_doi.append((o_ke, chi_phi + 1))

    return float('inf')  # khong co duong di


def _bfs_lay_duong_di(grid, diem_dau, diem_dich):
    """BFS tra ve ca duong di (list cac o)"""
    hang_doi = deque()
    hang_doi.append(diem_dau)

    den_tu = {}
    den_tu[diem_dau] = None

    while hang_doi:
        hien_tai = hang_doi.popleft()

        if hien_tai == diem_dich:
            # Truy vet duong di
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


def _tinh_tong_chi_phi_thu_tu(grid, vi_tri_dau, thu_tu_mau, vi_tri_base):
    """
    Tinh tong chi phi neu di theo thu tu thu mau cu the
    Dung BFS de tinh chi phi thuc (khong phai manhattan)
    """
    # Tao danh sach cac diem can di qua
    cac_diem = [vi_tri_dau]
    for mau in thu_tu_mau:
        cac_diem.append(mau)
    cac_diem.append(vi_tri_base)

    tong = 0
    for i in range(len(cac_diem) - 1):
        chi_phi_doan = _bfs_tinh_chi_phi(grid, cac_diem[i], cac_diem[i + 1])
        tong += chi_phi_doan

    return tong


def _xay_ket_qua(grid, vi_tri_dau, thu_tu_mau, vi_tri_base):
    """
    Sau khi co thu tu tot nhat, xay dung ket qua day du
    Di qua tung mau theo thu tu, dung BFS tim duong
    """
    duong_di_day_du = [vi_tri_dau]
    tong_buoc = 0
    tong_chi_phi = 0
    vi_tri_hien_tai = vi_tri_dau

    # Di qua tung mau
    cac_diem_can_den = list(thu_tu_mau) + [vi_tri_base]

    for diem_dich in cac_diem_can_den:
        duong_di = _bfs_lay_duong_di(grid, vi_tri_hien_tai, diem_dich)

        if not duong_di:
            return {
                "path": [], "visited": [],
                "stats": {"steps": 0, "cost": 0, "visited": 0, "found": False}
            }

        duong_di_day_du.extend(duong_di[1:])
        tong_buoc += len(duong_di) - 1
        for o in duong_di[1:]:
            tong_chi_phi += get_cost(grid, o)
        vi_tri_hien_tai = diem_dich

    return {
        "path":    duong_di_day_du,
        "visited": duong_di_day_du,  # local search: visited = path
        "stats": {
            "steps":   tong_buoc,
            "cost":    tong_chi_phi,
            "visited": tong_buoc,
            "found":   True,
        }
    }


# =====================================================
# HILL CLIMBING
# =====================================================

def hill_climbing(grid, start, samples, base):
    """
    Hill Climbing: Leo doc

    Cach hoat dong:
    1. Bat dau voi mot thu tu ngau nhien
    2. Thu hoan vi 2 mau bat ky
    3. Neu thu tu moi tot hon (chi phi it hon) → giu lai
    4. Lap lai cho den khi khong the cai thien them

    Nhuoc diem: Co the bi mac ket tai local optimum (dinh cuc bo)
    """
    # Bat dau voi thu tu ngau nhien
    thu_tu_hien_tai = list(samples)
    random.shuffle(thu_tu_hien_tai)

    chi_phi_hien_tai = _tinh_tong_chi_phi_thu_tu(grid, start, thu_tu_hien_tai, base)

    print(f"[HC] Bat dau voi chi phi: {chi_phi_hien_tai}")

    # Lap cho den khi khong cai thien duoc
    so_vong_lap = 0
    co_cai_thien = True

    while co_cai_thien:
        co_cai_thien = False
        so_vong_lap += 1

        # Thu tat ca cap hoan vi (i, j) co the
        for i in range(len(thu_tu_hien_tai)):
            for j in range(i + 1, len(thu_tu_hien_tai)):
                # Hoan vi vi tri i va j
                thu_tu_moi = list(thu_tu_hien_tai)
                thu_tu_moi[i], thu_tu_moi[j] = thu_tu_moi[j], thu_tu_moi[i]

                chi_phi_moi = _tinh_tong_chi_phi_thu_tu(grid, start, thu_tu_moi, base)

                if chi_phi_moi < chi_phi_hien_tai:
                    # Tim thay phuong an tot hon!
                    thu_tu_hien_tai  = thu_tu_moi
                    chi_phi_hien_tai = chi_phi_moi
                    co_cai_thien     = True

    print(f"[HC] Ket thuc sau {so_vong_lap} vong, chi phi cuoi: {chi_phi_hien_tai}")
    return _xay_ket_qua(grid, start, thu_tu_hien_tai, base)


# =====================================================
# SIMULATED ANNEALING (SA)
# =====================================================

def simulated_annealing(grid, start, samples, base):
    """
    Simulated Annealing: Lua phong nhiet

    Cach hoat dong:
    - Giong Hill Climbing NHUNG co them xac suat chap nhan nghiem XAU hon
    - Xac suat chap nhan: P = e^(-delta_E / T)
      + delta_E = chi phi moi - chi phi hien tai (duong neu te hon)
      + T = nhiet do (giam dan theo thoi gian)
    - Nhiet do cao → chap nhan nghiem xau nhieu (explore)
    - Nhiet do thap → chi chap nhan nghiem tot (exploit)
    - Tranh bi mac ket tai local optimum (khac Hill Climbing)
    """
    # Khoi tao
    thu_tu_hien_tai = list(samples)
    random.shuffle(thu_tu_hien_tai)

    chi_phi_hien_tai = _tinh_tong_chi_phi_thu_tu(grid, start, thu_tu_hien_tai, base)

    # Luu ket qua tot nhat tim duoc
    thu_tu_tot_nhat  = list(thu_tu_hien_tai)
    chi_phi_tot_nhat = chi_phi_hien_tai

    # Tham so nhiet do
    nhiet_do_ban_dau = 1000.0  # nhiet do cao ban dau
    nhiet_do_dung    = 0.1     # dung khi nhiet do qua thap
    he_so_lam_nguoi  = 0.995   # moi buoc giam nhiet do xuong 0.5%

    nhiet_do_hien_tai = nhiet_do_ban_dau
    so_buoc = 0

    print(f"[SA] Bat dau: chi phi = {chi_phi_hien_tai}, T = {nhiet_do_ban_dau}")

    # Vong lap chinh
    while nhiet_do_hien_tai > nhiet_do_dung:
        so_buoc += 1

        # Chon ngau nhien 2 vi tri de hoan vi
        i, j = random.sample(range(len(thu_tu_hien_tai)), 2)

        # Tao thu tu moi bang cach hoan vi i va j
        thu_tu_moi = list(thu_tu_hien_tai)
        thu_tu_moi[i], thu_tu_moi[j] = thu_tu_moi[j], thu_tu_moi[i]

        chi_phi_moi = _tinh_tong_chi_phi_thu_tu(grid, start, thu_tu_moi, base)

        # Tinh su thay doi chi phi
        delta = chi_phi_moi - chi_phi_hien_tai

        if delta < 0:
            # Nghiem moi tot hon → chap nhan ngay
            thu_tu_hien_tai  = thu_tu_moi
            chi_phi_hien_tai = chi_phi_moi
        else:
            # Nghiem moi kem hon → chap nhan voi xac suat e^(-delta/T)
            xac_suat = math.exp(-delta / nhiet_do_hien_tai)
            if random.random() < xac_suat:
                thu_tu_hien_tai  = thu_tu_moi
                chi_phi_hien_tai = chi_phi_moi

        # Cap nhat nghiem tot nhat
        if chi_phi_hien_tai < chi_phi_tot_nhat:
            thu_tu_tot_nhat  = list(thu_tu_hien_tai)
            chi_phi_tot_nhat = chi_phi_hien_tai

        # Lam nguoi nhiet do
        nhiet_do_hien_tai *= he_so_lam_nguoi

    print(f"[SA] Ket thuc sau {so_buoc} buoc, chi phi tot nhat: {chi_phi_tot_nhat}")
    return _xay_ket_qua(grid, start, thu_tu_tot_nhat, base)
