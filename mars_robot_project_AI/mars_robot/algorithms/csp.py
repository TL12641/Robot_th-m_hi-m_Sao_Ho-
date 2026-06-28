# csp.py
# Nhom 6: Constraint Satisfaction Problem (CSP)
# Gom: Backtracking va Forward Checking
#
# Bai toan CSP o day:
#   - Bien (Variable): thu tu thu mau (vi tri 1, 2, 3, 4, 5, 6)
#   - Domain: cac mau chua duoc gan
#   - Rang buoc: tong chi phi phai nho hon chi phi tot nhat tim duoc
#
# Khac Local Search:
#   - Local Search: cai thien dan
#   - CSP: thu het moi kha nang, cat nhanh neu vuot ngan sach

from collections import deque
from utils import get_neighbors, manhattan, get_cost


# =====================================================
# HAM BFS NOI BO
# =====================================================

def _bfs_chi_phi(grid, diem_dau, diem_dich):
    """BFS tinh chi phi thuc tu diem_dau den diem_dich"""
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

    return float('inf')


def _bfs_duong_di(grid, diem_dau, diem_dich):
    """BFS tra ve duong di day du"""
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


def _xay_ket_qua(grid, vi_tri_dau, thu_tu_mau, vi_tri_base):
    """Xay dung ket qua day du tu thu tu mau da chon"""
    duong_di_tong = [vi_tri_dau]
    so_buoc = 0
    chi_phi = 0
    vi_tri_ht = vi_tri_dau

    for diem_dich in list(thu_tu_mau) + [vi_tri_base]:
        duong = _bfs_duong_di(grid, vi_tri_ht, diem_dich)
        if not duong:
            return {"path": [], "visited": [],
                    "stats": {"steps": 0, "cost": 0, "visited": 0, "found": False}}
        duong_di_tong.extend(duong[1:])
        so_buoc += len(duong) - 1
        for o in duong[1:]:
            chi_phi += get_cost(grid, o)
        vi_tri_ht = diem_dich

    return {
        "path":    duong_di_tong,
        "visited": duong_di_tong,
        "stats":   {"steps": so_buoc, "cost": chi_phi,
                    "visited": so_buoc, "found": True}
    }


# =====================================================
# BACKTRACKING
# =====================================================

def _backtrack(grid, diem_dau, cac_mau_con_lai, vi_tri_base,
               vi_tri_hien_tai, chi_phi_tich_luy, ket_qua_tot_nhat):
    """
    Ham de quy backtracking:
    - Thu tung mau chua duoc chon
    - Cat nhanh neu chi phi da vuot ket qua tot nhat
    - Cap nhat ket_qua_tot_nhat khi tim duoc nghiem hoan toan

    ket_qua_tot_nhat la dict {'cost': ..., 'order': [...]}
    Dung dict de co the sua trong ham de quy (Python pass by reference voi dict)
    """
    if not cac_mau_con_lai:
        # Da gan het mau, tinh them chi phi ve base
        chi_phi_ve_base = _bfs_chi_phi(grid, vi_tri_hien_tai, vi_tri_base)
        tong = chi_phi_tich_luy + chi_phi_ve_base

        if tong < ket_qua_tot_nhat['cost']:
            ket_qua_tot_nhat['cost']  = tong
            ket_qua_tot_nhat['order'] = list(ket_qua_tot_nhat['duong_hien_tai'])
            print(f"[BT] Tim thay nghiem moi tot hon: chi phi = {tong}")
        return

    for mau in cac_mau_con_lai:
        # Tinh chi phi di den mau nay
        chi_phi_den_mau = _bfs_chi_phi(grid, vi_tri_hien_tai, mau)

        # Cat nhanh: neu chi phi hien tai da >= ket qua tot nhat thi bo qua
        if chi_phi_tich_luy + chi_phi_den_mau >= ket_qua_tot_nhat['cost']:
            continue  # cat nhanh!

        # Uoc tinh chi phi con lai (lower bound)
        cac_mau_moi = [m for m in cac_mau_con_lai if m != mau]
        uoc_tinh_con_lai = 0
        if cac_mau_moi:
            # Moi mau con lai tinh khoang cach Manhattan ngan nhat den mau khac hoac base
            for m in cac_mau_moi:
                cac_dich_co_the = cac_mau_moi + [vi_tri_base]
                uoc_tinh_con_lai += min(manhattan(m, d) for d in cac_dich_co_the if d != m)

        if chi_phi_tich_luy + chi_phi_den_mau + uoc_tinh_con_lai >= ket_qua_tot_nhat['cost']:
            continue  # cat nhanh bang lower bound

        # Gan mau nay vao thu tu
        ket_qua_tot_nhat['duong_hien_tai'].append(mau)

        # Goi de quy voi mau vua chon
        _backtrack(
            grid,
            diem_dau,
            cac_mau_moi,  # cac mau chua duoc chon
            vi_tri_base,
            mau,          # vi tri hien tai moi
            chi_phi_tich_luy + chi_phi_den_mau,
            ket_qua_tot_nhat
        )

        # Quay lui: bo mau vua them
        ket_qua_tot_nhat['duong_hien_tai'].pop()


def backtracking(grid, start, samples, base):
    """
    Backtracking CSP: thu moi thu tu co the, cat nhanh khi can
    Dam bao tim duoc thu tu TOI UU (tot nhat trong moi truong hop)
    """
    # Khoi tao nghiem ban dau bang greedy (de co upper bound ban dau)
    thu_tu_greedy = []
    cac_mau = list(samples)
    vi_tri = start

    while cac_mau:
        gan_nhat = min(cac_mau, key=lambda m: _bfs_chi_phi(grid, vi_tri, m))
        thu_tu_greedy.append(gan_nhat)
        cac_mau.remove(gan_nhat)
        vi_tri = gan_nhat

    # Tinh chi phi greedy lam upper bound ban dau
    tat_ca_diem = [start] + thu_tu_greedy
    chi_phi_greedy = sum(
        _bfs_chi_phi(grid, tat_ca_diem[i], tat_ca_diem[i + 1])
        for i in range(len(tat_ca_diem) - 1)
    )
    chi_phi_greedy += _bfs_chi_phi(grid, thu_tu_greedy[-1], base)

    print(f"[BT] Chi phi greedy ban dau: {chi_phi_greedy}")

    # Chay backtracking
    ket_qua = {
        'cost':            chi_phi_greedy,
        'order':           list(thu_tu_greedy),
        'duong_hien_tai':  []  # buffer tam thoi trong qua trinh de quy
    }

    _backtrack(grid, start, list(samples), base, start, 0, ket_qua)

    print(f"[BT] Chi phi toi uu: {ket_qua['cost']}")
    return _xay_ket_qua(grid, start, ket_qua['order'], base)


# =====================================================
# FORWARD CHECKING
# =====================================================

def _forward_check(grid, diem_dau, gan_cac_mau, domain_dict,
                   vi_tri_base, vi_tri_ht, chi_phi_tich_luy, ket_qua_tot_nhat):
    """
    De quy Forward Checking:
    - Giong Backtracking NHUNG sau moi buoc gan → kiem tra domain con lai
    - Neu domain cua bat ky bien nao trong (empty) → cat nhanh som
    """
    # Cac mau chua duoc gan
    chua_gan = [m for m in domain_dict if m not in gan_cac_mau]

    if not chua_gan:
        # Tat ca da duoc gan
        chi_phi_ve_base = _bfs_chi_phi(grid, vi_tri_ht, vi_tri_base)
        tong = chi_phi_tich_luy + chi_phi_ve_base
        if tong < ket_qua_tot_nhat['cost']:
            ket_qua_tot_nhat['cost']  = tong
            ket_qua_tot_nhat['order'] = list(gan_cac_mau)
            print(f"[FC] Nghiem moi: chi phi = {tong}")
        return

    # Chon bien theo MRV (binh thuong chon o gan nhat)
    bien_chon = min(chua_gan, key=lambda m: manhattan(vi_tri_ht, m))

    for gia_tri in list(domain_dict[bien_chon]):
        chi_phi_den = _bfs_chi_phi(grid, vi_tri_ht, gia_tri)

        if chi_phi_tich_luy + chi_phi_den >= ket_qua_tot_nhat['cost']:
            continue  # cat nhanh

        # Forward checking: cap nhat domain cac bien con lai
        domain_moi = {}
        for bien in domain_dict:
            if bien not in gan_cac_mau and bien != gia_tri:
                # Xoa gia_tri khoi domain cua bien nay
                domain_moi[bien] = set(domain_dict[bien])
                domain_moi[bien].discard(gia_tri)
            elif bien == gia_tri:
                domain_moi[bien] = set()
            else:
                domain_moi[bien] = set(domain_dict[bien])

        # Kiem tra domain empty (forward checking)
        co_domain_trong = False
        for bien in chua_gan:
            if bien != gia_tri and len(domain_moi.get(bien, {1})) == 0:
                co_domain_trong = True
                break

        if co_domain_trong:
            continue  # cat nhanh! Co bien khong con domain nao

        # Gan bien_chon = gia_tri va tiep tuc de quy
        gan_cac_mau.append(gia_tri)
        _forward_check(
            grid, diem_dau,
            gan_cac_mau, domain_moi,
            vi_tri_base, gia_tri,
            chi_phi_tich_luy + chi_phi_den,
            ket_qua_tot_nhat
        )
        gan_cac_mau.pop()


def forward_checking(grid, start, samples, base):
    """
    Forward Checking: Backtracking + kiem tra domain sau moi buoc gan
    Hieu qua hon Backtracking vi phat hien som cac nhanh khong kha thi
    """
    # Khoi tao domain: moi bien co the nhan gia tri la bat ki mau nao
    domain_dict = {}
    for mau in samples:
        domain_dict[mau] = set(samples)  # ban dau co the chon bat ki

    # Lay upper bound bang greedy
    thu_tu_greedy = []
    cac_mau = list(samples)
    vi_tri = start
    while cac_mau:
        gan_nhat = min(cac_mau, key=lambda m: _bfs_chi_phi(grid, vi_tri, m))
        thu_tu_greedy.append(gan_nhat)
        cac_mau.remove(gan_nhat)
        vi_tri = gan_nhat

    tat_ca_diem = [start] + thu_tu_greedy
    chi_phi_greedy = sum(
        _bfs_chi_phi(grid, tat_ca_diem[i], tat_ca_diem[i + 1])
        for i in range(len(tat_ca_diem) - 1)
    )
    chi_phi_greedy += _bfs_chi_phi(grid, thu_tu_greedy[-1], base)

    print(f"[FC] Upper bound greedy: {chi_phi_greedy}")

    ket_qua = {'cost': chi_phi_greedy, 'order': list(thu_tu_greedy)}

    _forward_check(grid, start, [], domain_dict, base, start, 0, ket_qua)

    print(f"[FC] Chi phi toi uu: {ket_qua['cost']}")
    return _xay_ket_qua(grid, start, ket_qua['order'], base)
