# nondeterministic.py
# Nhom 4: Thuat toan KHONG XAC DINH (Non-deterministic Search)
# Gom: AND-OR Search va Sensorless Search
#
# Diem khac biet voi cac nhom truoc:
#   AND-OR: Moi truong CO THE xay ra nhieu ket qua khi thuc hien hanh dong
#           → Can ke hoach du phong (Plan B) khi co su co
#   Sensorless: Robot KHONG BIET vi tri ban dau cua minh
#               → Can thu hep "belief state" den khi xac dinh duoc vi tri

import heapq
from collections import deque
from utils import get_neighbors, manhattan, get_cost, DIRECTIONS, GRID_SIZE, WALL, ROUGH


# =====================================================
# HAM A* NOI BO (dung cho ca 2 thuat toan)
# =====================================================

def _astar_noi_bo(grid, diem_dau, diem_dich):
    """
    A* don gian tra ve (duong_di, danh_sach_da_xet)
    Dung noi bo cho AND-OR va Sensorless
    """
    heap = []
    heapq.heappush(heap, (manhattan(diem_dau, diem_dich), 0, diem_dau))

    den_tu  = {diem_dau: None}
    g_score = {diem_dau: 0}
    da_xet  = []

    while heap:
        f, g, hien_tai = heapq.heappop(heap)

        if hien_tai in da_xet:
            continue
        da_xet.append(hien_tai)

        if hien_tai == diem_dich:
            duong_di = []
            o = diem_dich
            while o is not None:
                duong_di.append(o)
                o = den_tu[o]
            duong_di.reverse()
            return duong_di, da_xet

        for o_ke in get_neighbors(grid, hien_tai):
            g_moi = g + get_cost(grid, o_ke)
            if o_ke not in g_score or g_moi < g_score[o_ke]:
                g_score[o_ke] = g_moi
                den_tu[o_ke]  = hien_tai
                f_moi = g_moi + manhattan(o_ke, diem_dich)
                heapq.heappush(heap, (f_moi, g_moi, o_ke))

    return [], da_xet


# =====================================================
# AND-OR SEARCH
# =====================================================

def and_or_search(grid, start, samples, base, slip_prob=0.3):
    """
    AND-OR Search: Tim ke hoach co dieu kien cho moi truong bat dinh

    Moi truong bat dinh o day:
    - Khi robot di vao o ROUGH:
        (1 - slip_prob): di chuyen dung huong (Plan A - binh thuong)
        slip_prob: truot sang o ke ngau nhien (Plan B - phuc hoi)

    Ket qua tra ve:
    - path: duong di chinh (Plan A)
    - plan_b_segments: cac doan phuc hoi neu truot (Plan B, ve mau cam)
    - slip_prob: xac suat truot (duoc luu lai de hien thi)

    Cach xay Plan B:
    - Voi moi o ROUGH tren duong di chinh
    - Tim cac o co the truot den
    - Xay ke hoach phuc hoi: tu o truot → muc tieu goc
    """
    # Chon thu tu thu mau: mau gan nhat truoc (Greedy Nearest)
    thu_tu = []
    con_lai = list(samples)
    vi_tri  = start

    while con_lai:
        gan_nhat = min(con_lai, key=lambda s: manhattan(vi_tri, s))
        thu_tu.append(gan_nhat)
        con_lai.remove(gan_nhat)
        vi_tri = gan_nhat
    thu_tu.append(base)

    # Bien luu toan bo ket qua
    duong_di_tong  = [start]
    tat_ca_da_xet  = []
    ke_hoach_b     = []     # cac doan Plan B (de ve animation)
    tong_buoc      = 0
    tong_chi_phi   = 0
    vi_tri_hien_tai = start

    for muc_tieu in thu_tu:
        # Tim duong chinh bang A*
        duong_di, da_xet = _astar_noi_bo(grid, vi_tri_hien_tai, muc_tieu)

        if not duong_di:
            print(f"[AND-OR] Khong tim duoc duong den {muc_tieu}")
            return {
                "path": [], "visited": tat_ca_da_xet,
                "plan_b_segments": [],
                "stats": {"steps": 0, "cost": 0,
                          "visited": len(tat_ca_da_xet), "found": False}
            }

        duong_di_tong.extend(duong_di[1:])
        tat_ca_da_xet.extend(da_xet)
        tong_buoc    += len(duong_di) - 1
        for o in duong_di[1:]:
            tong_chi_phi += get_cost(grid, o)

        # --- Xay Plan B: xu ly truong hop truot tren ROUGH ---
        for i in range(len(duong_di) - 1):
            o_tiep_theo = duong_di[i + 1]

            # Chi quan tam neu o tiep theo la ROUGH
            if grid[o_tiep_theo[0]][o_tiep_theo[1]] == ROUGH:
                o_hien_tai = duong_di[i]

                # Tim cac o co the truot den (khac o tiep theo chinh)
                cac_o_ke = get_neighbors(grid, o_hien_tai)
                o_co_the_truot = []
                for o_ke in cac_o_ke:
                    if o_ke != o_tiep_theo:
                        o_co_the_truot.append(o_ke)

                # Xay ke hoach phuc hoi cho TAT CA huong co the truot
                for o_truot in o_co_the_truot:
                    duong_phuc_hoi, _ = _astar_noi_bo(grid, o_truot, muc_tieu)
                    if duong_phuc_hoi and len(duong_phuc_hoi) > 1:
                        ke_hoach_b.append(duong_phuc_hoi)

        vi_tri_hien_tai = muc_tieu

    print(f"[AND-OR] Hoan thanh: {tong_buoc} buoc, {len(ke_hoach_b)} ke hoach phuc hoi")

    return {
        "path":            duong_di_tong,
        "visited":         tat_ca_da_xet,
        "plan_b_segments": ke_hoach_b,
        "slip_prob":       slip_prob,
        "stats": {
            "steps":        tong_buoc,
            "cost":         tong_chi_phi,
            "visited":      len(tat_ca_da_xet),
            "found":        True,
            "plan_b_count": len(ke_hoach_b),
        }
    }


# =====================================================
# SENSORLESS SEARCH
# =====================================================

def _ap_dung_hanh_dong(grid, belief_state, d_hang, d_cot):
    """
    Ap dung hanh dong (d_hang, d_cot) len toan bo belief state
    Neu mot vi tri trong belief khong di duoc → giu nguyen vi tri do
    """
    belief_moi = set()

    for hang, cot in belief_state:
        hang_moi = hang + d_hang
        cot_moi  = cot  + d_cot

        # Kiem tra co di duoc khong
        if (0 <= hang_moi < GRID_SIZE and
                0 <= cot_moi < GRID_SIZE and
                grid[hang_moi][cot_moi] != WALL):
            belief_moi.add((hang_moi, cot_moi))
        else:
            belief_moi.add((hang, cot))  # khong di duoc → giu nguyen

    return frozenset(belief_moi)


def sensorless_search(grid, start, samples, base, **kwargs):
    """
    Sensorless Search: Robot KHONG BIET vi tri ban dau

    Cach hoat dong:
    1. Belief state ban dau = TAT CA cac o khong phai tuong
       (robot co the dang o bat ki dau)
    2. Moi buoc chon hanh dong tot nhat de thu hep belief state
       → Chon hanh dong giam kich thuoc belief nhieu nhat
    3. Khi belief state giam xuong nho → coi nhu biet vi tri
    4. Tu vi tri da xac dinh → thu mau va ve base binh thuong

    Ky thuat chon hanh dong: Greedy - moi buoc chon huong thu hep belief nhieu nhat
    """
    # --- Buoc 1: Khoi tao belief state ---
    belief_ban_dau = set()
    for hang in range(GRID_SIZE):
        for cot in range(GRID_SIZE):
            if grid[hang][cot] != WALL:
                belief_ban_dau.add((hang, cot))

    belief_ban_dau = frozenset(belief_ban_dau)
    print(f"[SL] Belief ban dau: {len(belief_ban_dau)} vi tri kha thi")

    # --- Buoc 2: Greedy thu hep belief state ---
    belief_hien_tai    = belief_ban_dau
    lich_su_belief     = [belief_ban_dau]   # luu lich su de ve animation
    chuoi_hanh_dong    = []
    da_tham_belief     = {belief_ban_dau}

    # Gioi han so buoc tranh vo han vong lap
    GIOI_HAN_BUOC = 35

    for buoc in range(GIOI_HAN_BUOC):
        if len(belief_hien_tai) <= 1:
            break  # da xac dinh vi tri!

        # Tim hanh dong tot nhat (giam belief nhieu nhat)
        hanh_dong_tot_nhat = None
        belief_tot_nhat    = belief_hien_tai
        diem_so_tot_nhat   = float('inf')

        for d_hang, d_cot in DIRECTIONS:
            belief_moi = _ap_dung_hanh_dong(grid, belief_hien_tai, d_hang, d_cot)

            # Bo qua neu da tham belief nay roi (tranh lap vong)
            if belief_moi in da_tham_belief or belief_moi == belief_hien_tai:
                continue

            # Tinh diem so: uu tien belief nho + huong ve base
            trung_tam_hang = sum(p[0] for p in belief_moi) / len(belief_moi)
            trung_tam_cot  = sum(p[1] for p in belief_moi) / len(belief_moi)
            khoang_den_base = manhattan(
                (int(trung_tam_hang), int(trung_tam_cot)), base
            )
            diem_so = len(belief_moi) * 5 + khoang_den_base

            if diem_so < diem_so_tot_nhat:
                diem_so_tot_nhat   = diem_so
                hanh_dong_tot_nhat = (d_hang, d_cot)
                belief_tot_nhat    = belief_moi

        if hanh_dong_tot_nhat is None:
            break  # khong co hanh dong nao giup thu hep belief

        # Ap dung hanh dong
        belief_hien_tai = belief_tot_nhat
        da_tham_belief.add(belief_hien_tai)
        chuoi_hanh_dong.append(hanh_dong_tot_nhat)
        lich_su_belief.append(belief_hien_tai)

    print(f"[SL] Sau {len(chuoi_hanh_dong)} hanh dong, belief con: {len(belief_hien_tai)} vi tri")

    # --- Buoc 3: Xac dinh vi tri (chon o trong belief gan start nhat) ---
    vi_tri_xac_dinh = min(
        belief_hien_tai,
        key=lambda p: manhattan(p, start)
    ) if belief_hien_tai else start

    # --- Buoc 4: Simulate di chuyen thuc te tu start ---
    vi_tri_sim = start
    duong_sensorless = [start]

    for d_hang, d_cot in chuoi_hanh_dong:
        hang_moi = vi_tri_sim[0] + d_hang
        cot_moi  = vi_tri_sim[1] + d_cot

        if (0 <= hang_moi < GRID_SIZE and
                0 <= cot_moi < GRID_SIZE and
                grid[hang_moi][cot_moi] != WALL):
            vi_tri_sim = (hang_moi, cot_moi)

        duong_sensorless.append(vi_tri_sim)

    # --- Buoc 5: Thu mau va ve base tu vi tri xac dinh ---
    cac_mau = list(samples)
    vi_tri_ht = vi_tri_sim
    tong_buoc  = len(chuoi_hanh_dong)
    tong_cp    = sum(get_cost(grid, p) for p in duong_sensorless[1:])

    while cac_mau:
        gan_nhat = min(cac_mau, key=lambda m: manhattan(vi_tri_ht, m))
        cac_mau.remove(gan_nhat)

        duong, _ = _astar_noi_bo(grid, vi_tri_ht, gan_nhat)
        if duong:
            duong_sensorless.extend(duong[1:])
            tong_buoc += len(duong) - 1
            for o in duong[1:]:
                tong_cp += get_cost(grid, o)
            vi_tri_ht = gan_nhat

    # Ve base
    duong_base, _ = _astar_noi_bo(grid, vi_tri_ht, base)
    if duong_base:
        duong_sensorless.extend(duong_base[1:])
        tong_buoc += len(duong_base) - 1
        for o in duong_base[1:]:
            tong_cp += get_cost(grid, o)

    return {
        "path":           duong_sensorless,
        "visited":        list(belief_ban_dau),  # hien thi belief ban dau
        "belief_history": lich_su_belief,         # lich su thu hep belief
        "stats": {
            "steps":        tong_buoc,
            "cost":         tong_cp,
            "visited":      len(belief_ban_dau),
            "found":        True,
            "belief_init":  len(belief_ban_dau),
            "belief_final": len(belief_hien_tai),
        }
    }
