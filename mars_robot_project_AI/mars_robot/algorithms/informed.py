# informed.py
# Nhom 2: Thuat toan tim kiem CO THONG TIN (Informed Search)
# Gom: Greedy Best-First va A* (A-Star)
# Khac nhom 1: su dung them HEURISTIC de uoc tinh khoang cach den dich

import heapq
from utils import get_neighbors, manhattan, get_cost


# =====================================================
# HAM DUNG CHUNG
# =====================================================

def _truy_vet(den_tu, diem_dau, diem_dich):
    """Truy nguoc duong di tu dich → dau, tra ve list tu dau → dich"""
    duong_di = []
    hien_tai = diem_dich
    while hien_tai != diem_dau:
        duong_di.append(hien_tai)
        hien_tai = den_tu[hien_tai]
    duong_di.append(diem_dau)
    duong_di.reverse()
    return duong_di


# =====================================================
# GREEDY BEST-FIRST SEARCH
# =====================================================

def _tim_duong_greedy(grid, diem_dau, diem_dich):
    """
    Greedy Best-First Search:
    - Moi buoc chon o co h(n) nho nhat (h = manhattan den dich)
    - Nhanh hon A* nhung khong dam bao toi uu
    - Co the bi "mac ket" neu huong Manhattan khong thuc su la ngan nhat

    Su dung priority queue (hang doi uu tien)
    """
    # Hang doi uu tien: (gia_tri_h, vi_tri)
    hang_doi_uu_tien = []
    h_dau = manhattan(diem_dau, diem_dich)
    heapq.heappush(hang_doi_uu_tien, (h_dau, diem_dau))

    den_tu = {}
    den_tu[diem_dau] = None

    da_duyet = []  # danh sach theo thu tu duyet

    while hang_doi_uu_tien:
        # Lay o co h nho nhat
        _, hien_tai = heapq.heappop(hang_doi_uu_tien)

        if hien_tai in da_duyet:
            continue  # bo qua neu da duyet roi

        da_duyet.append(hien_tai)

        if hien_tai == diem_dich:
            duong_di = _truy_vet(den_tu, diem_dau, diem_dich)
            return duong_di, da_duyet

        # Mo rong: them cac o ke vao hang doi
        for o_ke in get_neighbors(grid, hien_tai):
            if o_ke not in den_tu:
                den_tu[o_ke] = hien_tai
                h_ke = manhattan(o_ke, diem_dich)  # heuristic cua o ke
                heapq.heappush(hang_doi_uu_tien, (h_ke, o_ke))

    return [], da_duyet


# =====================================================
# A* SEARCH
# =====================================================

def _tim_duong_astar(grid, diem_dau, diem_dich):
    """
    A* (A-Star):
    - Cong thuc: f(n) = g(n) + h(n)
        + g(n) = chi phi thuc te tu dau den n (tinh ca ROUGH x5)
        + h(n) = uoc tinh den dich (dung Manhattan)
    - Thuat toan TOI UU: luon tim duong co chi phi nho nhat
    - Cham hon Greedy nhung ket qua tot hon (dac biet khi co ROUGH)

    Su dung priority queue theo gia tri f
    """
    # Heap: (f, g, vi_tri)
    # Can them g de tranh so sanh tuple khi f bang nhau
    heap = []
    f_dau = manhattan(diem_dau, diem_dich)
    heapq.heappush(heap, (f_dau, 0, diem_dau))

    den_tu  = {diem_dau: None}
    g_score = {diem_dau: 0}  # chi phi thuc te den tung o

    da_duyet = []

    while heap:
        f, g, hien_tai = heapq.heappop(heap)

        if hien_tai in da_duyet:
            continue

        da_duyet.append(hien_tai)

        if hien_tai == diem_dich:
            duong_di = _truy_vet(den_tu, diem_dau, diem_dich)
            return duong_di, da_duyet

        # Xet cac o ke
        for o_ke in get_neighbors(grid, hien_tai):
            # Chi phi di vao o ke (ROUGH = 5, con lai = 1)
            chi_phi_buoc = get_cost(grid, o_ke)
            g_moi = g + chi_phi_buoc

            # Cap nhat neu tim duoc duong tot hon
            if o_ke not in g_score or g_moi < g_score[o_ke]:
                g_score[o_ke] = g_moi
                den_tu[o_ke]  = hien_tai
                h_ke   = manhattan(o_ke, diem_dich)
                f_moi  = g_moi + h_ke
                heapq.heappush(heap, (f_moi, g_moi, o_ke))

    return [], da_duyet


# =====================================================
# HAM CHAY CHUNG (tuong tu uninformed.py)
# =====================================================

def _chay_thuat_toan(grid, diem_xuat_phat, danh_sach_mau, vi_tri_base, ham_tim_duong):
    """
    Chay thuat toan theo thu tu thu mau gan nhat truoc
    Tra ve ket qua chuan (path, visited, stats)
    """
    cac_mau_con_lai  = list(danh_sach_mau)
    vi_tri_hien_tai  = diem_xuat_phat

    duong_di_day_du = [diem_xuat_phat]
    tat_ca_da_xet   = []
    tong_so_buoc    = 0
    tong_chi_phi    = 0

    # Thu tung mau theo thu tu gan nhat
    while cac_mau_con_lai:
        mau_gan_nhat    = min(cac_mau_con_lai,
                              key=lambda s: manhattan(vi_tri_hien_tai, s))
        cac_mau_con_lai.remove(mau_gan_nhat)

        duong_di, da_xet = ham_tim_duong(grid, vi_tri_hien_tai, mau_gan_nhat)

        if not duong_di:
            print(f"[WARN] Khong tim duoc duong di den mau tai {mau_gan_nhat}")
            return {
                "path": [], "visited": tat_ca_da_xet,
                "stats": {"steps": 0, "cost": 0,
                          "visited": len(tat_ca_da_xet), "found": False}
            }

        duong_di_day_du.extend(duong_di[1:])
        tat_ca_da_xet.extend(da_xet)
        tong_so_buoc += len(duong_di) - 1

        for o in duong_di[1:]:
            tong_chi_phi += get_cost(grid, o)

        vi_tri_hien_tai = mau_gan_nhat

    # Ve base
    duong_ve, da_xet_base = ham_tim_duong(grid, vi_tri_hien_tai, vi_tri_base)
    if duong_ve:
        duong_di_day_du.extend(duong_ve[1:])
        tat_ca_da_xet.extend(da_xet_base)
        tong_so_buoc += len(duong_ve) - 1
        for o in duong_ve[1:]:
            tong_chi_phi += get_cost(grid, o)

    return {
        "path":    duong_di_day_du,
        "visited": tat_ca_da_xet,
        "stats": {
            "steps":   tong_so_buoc,
            "cost":    tong_chi_phi,
            "visited": len(tat_ca_da_xet),
            "found":   True,
        }
    }


# =====================================================
# HAM CONG KHAI
# =====================================================

def greedy(grid, start, samples, base):
    """
    Greedy Best-First: chon huong co heuristic tot nhat
    Nhanh nhung khong toi uu (co the bo qua ROUGH re hon)
    """
    return _chay_thuat_toan(grid, start, samples, base, _tim_duong_greedy)


def astar(grid, start, samples, base):
    """
    A*: Ket hop chi phi thuc te + heuristic
    Toi uu chi phi, tinh ca ROUGH x5
    """
    return _chay_thuat_toan(grid, start, samples, base, _tim_duong_astar)
