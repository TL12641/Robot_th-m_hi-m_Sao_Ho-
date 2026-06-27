# uninformed.py
# Nhom 1: Thuat toan tim kiem MU THONG TIN (Uninformed Search)
# Gom: BFS (Breadth-First Search) va DFS (Depth-First Search)
# Cac thuat toan nay khong dung them thong tin gi ve muc tieu

from collections import deque
from utils import get_neighbors, manhattan, get_cost


# =====================================================
# HAM DUNG CHUNG
# =====================================================

def truy_vet_duong_di(den_tu, diem_dau, diem_dich):
    """
    Truy vet nguoc tu diem_dich ve diem_dau
    Su dung dictionary den_tu[o] = o truoc do
    Tra ve list cac o tu dau -> dich
    """
    duong_di = []
    hien_tai = diem_dich

    # Truy nguoc tu dich ve dau
    while hien_tai != diem_dau:
        duong_di.append(hien_tai)
        hien_tai = den_tu[hien_tai]

    duong_di.append(diem_dau)  # them diem dau vao
    duong_di.reverse()         # dao nguoc lai (dau -> dich)
    return duong_di


# =====================================================
# BFS - Breadth First Search
# =====================================================

def _tim_duong_bfs(grid, diem_dau, diem_dich):
    """
    BFS: Tim duong ngan nhat tu diem_dau den diem_dich
    Su dung hang doi FIFO (First In First Out)

    Nguyen ly:
    - Duyet tat ca o o tang 1 truoc, roi moi den tang 2, v.v...
    - Dam bao tim duoc duong ngan nhat (so buoc it nhat)
    - Khong tinh den chi phi ROUGH (coi moi o co chi phi bang nhau)

    Returns: (duong_di, danh_sach_da_xet)
    """
    # Khoi tao hang doi voi diem bat dau
    hang_doi = deque()
    hang_doi.append(diem_dau)

    # Luu lai di chuyen tu o nao (de truy vet duong di sau)
    den_tu = {}
    den_tu[diem_dau] = None  # diem dau khong co o truoc

    # Luu thu tu cac o da duyet (de ve animation)
    thu_tu_duyet = []

    while hang_doi:
        hien_tai = hang_doi.popleft()  # lay o dau hang doi
        thu_tu_duyet.append(hien_tai)

        # Kiem tra da den dich chua
        if hien_tai == diem_dich:
            duong_di = truy_vet_duong_di(den_tu, diem_dau, diem_dich)
            return duong_di, thu_tu_duyet

        # Mo rong: them cac o ke vao hang doi
        for o_ke in get_neighbors(grid, hien_tai):
            if o_ke not in den_tu:  # chua tham o nay
                den_tu[o_ke] = hien_tai
                hang_doi.append(o_ke)

    # Khong tim duoc duong di
    return [], thu_tu_duyet


# =====================================================
# DFS - Depth First Search
# =====================================================

def _tim_duong_dfs(grid, diem_dau, diem_dich):
    """
    DFS: Tim duong di su dung ngan xep (stack) - di sau truoc

    Nguyen ly:
    - Uu tien di sau vao mot nhanh truoc khi quay lui
    - KHONG dam bao duong di ngan nhat
    - Tieu thu it bo nho hon BFS trong mot so truong hop

    NOTE: DFS co the cho ket qua khong toi uu!
    Returns: (duong_di, danh_sach_da_xet)
    """
    # Khoi tao ngan xep (dung list lam stack)
    ngan_xep = []
    ngan_xep.append(diem_dau)

    den_tu = {}
    den_tu[diem_dau] = None

    # Danh sach o da tham (tranh duyet lai)
    da_tham = []

    while ngan_xep:
        hien_tai = ngan_xep.pop()  # lay o tren cung ngan xep

        # Bo qua neu da tham roi
        if hien_tai in da_tham:
            continue

        da_tham.append(hien_tai)

        # Kiem tra dich
        if hien_tai == diem_dich:
            duong_di = truy_vet_duong_di(den_tu, diem_dau, diem_dich)
            return duong_di, da_tham

        # Mo rong: day cac o ke vao ngan xep
        for o_ke in get_neighbors(grid, hien_tai):
            if o_ke not in den_tu:
                den_tu[o_ke] = hien_tai
                ngan_xep.append(o_ke)

    return [], da_tham


# =====================================================
# HAM CHAY THUAT TOAN TREN NHIEM VU ROBOT
# =====================================================

def _chay_thuat_toan(grid, diem_xuat_phat, danh_sach_mau, vi_tri_base, ham_tim_duong):
    """
    Chay thuat toan tim duong theo thu tu: thu mau gan nhat truoc (greedy nearest)
    Sau khi thu het mau → ve base

    Tra ve dict chuan: path, visited, stats
    """
    # Cac bien theo doi hanh trinh
    cac_mau_con_lai = list(danh_sach_mau)  # copy de khong sua original
    vi_tri_hien_tai = diem_xuat_phat

    duong_di_day_du  = [diem_xuat_phat]
    tat_ca_da_xet    = []
    tong_so_buoc     = 0
    tong_chi_phi     = 0

    # --- Thu mau theo thu tu gan nhat ---
    while cac_mau_con_lai:
        # Tim mau gan nhat (dung manhattan de uoc tinh)
        mau_gan_nhat = None
        khoang_nho_nhat = float('inf')

        for mau in cac_mau_con_lai:
            khoang = manhattan(vi_tri_hien_tai, mau)
            if khoang < khoang_nho_nhat:
                khoang_nho_nhat = khoang
                mau_gan_nhat = mau

        cac_mau_con_lai.remove(mau_gan_nhat)

        # Tim duong tu vi tri hien tai den mau gan nhat
        duong_di, da_xet = ham_tim_duong(grid, vi_tri_hien_tai, mau_gan_nhat)

        if not duong_di:
            # Khong tim duoc duong → tra ve that bai
            print(f"[WARN] Khong tim duoc duong tu {vi_tri_hien_tai} den {mau_gan_nhat}")
            ket_qua = {
                "path":    [],
                "visited": tat_ca_da_xet,
                "stats": {
                    "steps":   0,
                    "cost":    0,
                    "visited": len(tat_ca_da_xet),
                    "found":   False,
                }
            }
            return ket_qua

        # Gop duong di (bo diem dau vi da co trong duong_di_day_du)
        duong_di_day_du.extend(duong_di[1:])
        tat_ca_da_xet.extend(da_xet)
        tong_so_buoc += len(duong_di) - 1

        # Tinh chi phi thuc (co tinh ROUGH x5)
        for o in duong_di[1:]:
            tong_chi_phi += get_cost(grid, o)

        vi_tri_hien_tai = mau_gan_nhat

    # --- Ve base sau khi thu xong het mau ---
    duong_di_ve_base, da_xet_base = ham_tim_duong(grid, vi_tri_hien_tai, vi_tri_base)

    if duong_di_ve_base:
        duong_di_day_du.extend(duong_di_ve_base[1:])
        tat_ca_da_xet.extend(da_xet_base)
        tong_so_buoc += len(duong_di_ve_base) - 1
        for o in duong_di_ve_base[1:]:
            tong_chi_phi += get_cost(grid, o)

    # Tra ve ket qua
    ket_qua = {
        "path":    duong_di_day_du,
        "visited": tat_ca_da_xet,
        "stats": {
            "steps":   tong_so_buoc,
            "cost":    tong_chi_phi,
            "visited": len(tat_ca_da_xet),
            "found":   True,
        }
    }
    return ket_qua


# =====================================================
# HAM CONG KHAI (Public API)
# =====================================================

def bfs(grid, start, samples, base):
    """
    BFS: Tim duong ngan nhat, duyet theo tung lop
    Khong tinh chi phi ROUGH, chi tinh so buoc
    """
    return _chay_thuat_toan(grid, start, samples, base, _tim_duong_bfs)


def dfs(grid, start, samples, base):
    """
    DFS: Tim duong bang ngan xep, di sau truoc
    Khong dam bao toi uu, co the tim duong dai hon BFS
    """
    return _chay_thuat_toan(grid, start, samples, base, _tim_duong_dfs)
