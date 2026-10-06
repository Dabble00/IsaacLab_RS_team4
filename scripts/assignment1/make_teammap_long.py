"""팀원 평가맵을 +x 로 한 벌 더 이어 붙인 긴 맵 만들기 (10/2 19:00 사용자: "지형 200 m 로, 그대로 복사해서 다시 돌려봐").
원본: ant_maps/generated/heldout_seed20261008_friction_spectrum.usd (타일 20행×10열 = x −80~80, y −40~40, 테두리 20 m)
결과: ant_maps/generated/heldout_seed20261008_friction_spectrum_x2.usd + .json (새 파일. 원본은 안 건드림)
  - 타일 영역의 모든 메시(보이는 전체 메시 1개 + 재질 그룹별 충돌 메시 37개)를 x 방향으로 +160 m 옮겨 한 벌 더 붙인다 → 타일 40행 = x −80~240
    (출발점 x −75~−10 에서 타일 끝까지 250~315 m: 16초에 180 m 가는 모델도 끝에 못 닿는다)
  - 테두리(평지 ring, 충돌 메시 'border' + 보이는 메시의 ring 면 8개)는 새 범위(x −100~260, y −60~60)로 다시 만든다 (겹치는 평면이 구덩이를 덮지 않게)
  - 메타데이터(USD customData + JSON): 행 40, terrain_origins·tile_types·tile_materials·tiles 를 복사본(행 +20, x +160)으로 늘림. 출발점 100개는 그대로.
    팀원 검증 코드(evaluation_metadata.py)가 보는 항목(출발점·평지 타일·SHA256)은 그대로 통과한다
사용법: PYTHONPATH=<usd-core 설치 경로> python RS_수업자료/3주차/tools/make_teammap_long.py   (Isaac Sim 없이 usd-core 로. 약 1분)
"""
import hashlib
import json
import os

import numpy as np
from pxr import Usd, UsdGeom, Vt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ant_maps", "generated")  # 저장소 기준 상대 경로 (깃허브 사본)
SRC = f"{ROOT}/heldout_seed20261008_friction_spectrum.usd"
DST = f"{ROOT}/heldout_seed20261008_friction_spectrum_x2.usd"
SHIFT = 160.0  # [m] 타일 20행 × 8 m
X0, X1, Y0, Y1 = -80.0, 80.0, -40.0, 40.0  # 원본 타일 영역
BW = 20.0  # 테두리 폭
NX1 = X1 + SHIFT  # 새 타일 끝 x = 240
EPS = 1e-6


def ring_geometry(x0, x1, y0, y1, bw):
    """타일 영역 둘레의 평지 ring (z = 0): 4개 사각형 = 삼각형 8개. 원본 border 메시와 같은 꼴"""
    quads = [
        ((x0 - bw, y0), (x0, y1)),  # 왼쪽 띠
        ((x1, y0), (x1 + bw, y1)),  # 오른쪽 띠
        ((x0 - bw, y1), (x1 + bw, y1 + bw)),  # 위 띠
        ((x0 - bw, y0 - bw), (x1 + bw, y0)),  # 아래 띠
    ]
    pts, idx = [], []
    for (ax, ay), (bx, by) in quads:
        base = len(pts)
        pts += [(ax, ay, 0.0), (ax, by, 0.0), (bx, ay, 0.0), (bx, by, 0.0)]
        idx += [base, base + 3, base + 1, base + 2, base + 3, base]  # 원본과 같은 감기 방향 (0,3,1), (2,3,0)
    return np.array(pts, dtype=np.float32), np.array(idx, dtype=np.int32)


def duplicate(pts, fvi, normals, keep_face, colors=None):
    """keep_face(F,) 가 참인 면만 남긴 뒤, 같은 면을 x+SHIFT 로 한 벌 더 붙인다. normals 는 점 단위 또는 면꼭짓점 단위 둘 다 처리.
    colors (10/3 12:00 추가): 점 단위 색 primvar(displayColor)도 점과 똑같이 복사한다 — 처음 버전은 이걸 빼먹어 긴 맵 색이 깨졌음 (물리·높이 센서는 영향 없음)"""
    tri = fvi.reshape(-1, 3)[keep_face]
    used = np.unique(tri)
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    p0 = pts[used]
    t0 = remap[tri]
    p1 = p0.copy()
    p1[:, 0] += SHIFT
    new_pts = np.concatenate([p0, p1])
    new_tri = np.concatenate([t0, t0 + len(p0)])
    new_colors = None
    if colors is not None and len(colors) == len(pts):
        c0 = colors[used]
        new_colors = np.concatenate([c0, c0])
    new_normals = None
    if normals is not None and len(normals) == len(pts):  # 점 단위
        n0 = normals[used]
        new_normals = np.concatenate([n0, n0])
    elif normals is not None and len(normals) == len(fvi):  # 면꼭짓점 단위
        n0 = normals.reshape(-1, 3, 3)[keep_face].reshape(-1, 3)
        new_normals = np.concatenate([n0, n0])
    return new_pts, new_tri.reshape(-1), new_normals, new_colors


def write_colors(prim, colors):
    """점 단위 displayColor / displayOpacity 를 새 점 수에 맞춰 다시 쓴다"""
    from pxr import UsdGeom as _UG

    pv = _UG.PrimvarsAPI(prim)
    c = pv.GetPrimvar("displayColor")
    if c and c.HasValue():
        c.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
    o = pv.GetPrimvar("displayOpacity")
    if o and o.HasValue():
        o.Set(Vt.FloatArray.FromNumpy(np.ones(len(colors), dtype=np.float32)))


def write_mesh(mesh, pts, fvi, normals):
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(fvi) // 3, 3, dtype=np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(fvi.astype(np.int32)))
    if normals is not None:
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    elif mesh.GetNormalsAttr().HasAuthoredValue():
        mesh.GetNormalsAttr().Clear()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([tuple(lo.astype(float)), tuple(hi.astype(float))]))


stage = Usd.Stage.Open(SRC)
ring_pts, ring_idx = ring_geometry(X0, NX1, Y0, Y1, BW)
report = []
for prim in Usd.PrimRange(stage.GetDefaultPrim()):
    if prim.GetTypeName() != "Mesh":
        continue
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    fvc = np.array(mesh.GetFaceVertexCountsAttr().Get())
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    assert np.all(fvc == 3), f"{prim.GetPath()} 삼각형이 아님"
    normals = mesh.GetNormalsAttr().Get()
    normals = np.array(normals, dtype=np.float64) if normals is not None and len(normals) > 0 else None
    path = str(prim.GetPath())
    if path.endswith("/colliders/border/mesh"):
        write_mesh(mesh, ring_pts, ring_idx, None)
        report.append(f"{path}: 테두리 ring 다시 만듦 (x {X0 - BW}~{NX1 + BW}, y {Y0 - BW}~{Y1 + BW})")
        continue
    P = pts[fvi.reshape(-1, 3)]  # (F,3,3)
    is_ring = ((np.abs(P[:, :, 0]) >= 100 - EPS) | (np.abs(P[:, :, 1]) >= 60 - EPS)).any(axis=1)  # 타일 면은 |x|=100, |y|=60 에 닿지 않음
    cpv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    colors = np.array(cpv.Get(), dtype=np.float64) if cpv and cpv.HasValue() and len(cpv.Get()) == len(pts) else None
    new_pts, new_fvi, new_normals, new_colors = duplicate(pts, fvi, normals, ~is_ring, colors)
    if is_ring.any():  # 보이는 전체 메시: ring 면도 들어 있음 → 새 ring 을 뒤에 붙임
        base = len(new_pts)
        new_pts = np.concatenate([new_pts, ring_pts])
        new_fvi = np.concatenate([new_fvi, ring_idx + base])
        if new_colors is not None:  # ring 색 = 원본 ring 꼭짓점(|x| = 100) 의 색
            ring_vertex = np.flatnonzero(np.abs(pts[:, 0]) >= 100 - EPS)
            ring_color = colors[ring_vertex[0]] if len(ring_vertex) else np.array([0.5, 0.5, 0.5])
            new_colors = np.concatenate([new_colors, np.tile(ring_color, (len(ring_pts), 1))])
        if new_normals is not None:
            if len(new_normals) == base:  # 점 단위
                new_normals = np.concatenate([new_normals, np.tile([0.0, 0.0, 1.0], (len(ring_pts), 1))])
            else:  # 면꼭짓점 단위
                new_normals = np.concatenate([new_normals, np.tile([0.0, 0.0, 1.0], (len(ring_idx), 1))])
    write_mesh(mesh, new_pts, new_fvi, new_normals)
    if new_colors is not None:
        write_colors(prim, new_colors)
    report.append(f"{path}: 면 {len(fvi) // 3} → {len(new_fvi) // 3} (ring 면 {int(is_ring.sum())}개 교체, 색 {'복사' if new_colors is not None else '없음'})")

# 메타데이터 늘리기 (USD customData + JSON)
md = json.loads(stage.GetDefaultPrim().GetCustomDataByKey("map_metadata"))
orig_json = json.load(open(SRC[:-4] + ".json", encoding="utf-8"))
rows = md["generator"]["rows"]
md["generator"]["rows"] = rows * 2
md["terrain_origins"] = md["terrain_origins"] + [[[o[0] + SHIFT, o[1], o[2]] for o in row] for row in md["terrain_origins"]]
md["tile_types"] = md["tile_types"] + [list(r) for r in md["tile_types"]]
md["tile_materials"] = md["tile_materials"] + [list(r) for r in md["tile_materials"]]
md["tiles"] = md["tiles"] + [dict(t, row=t["row"] + rows) for t in md["tiles"]]
md["long_map_note"] = f"x2: original 20 rows copied once more at x+{SHIFT:.0f} m (rows 20-39). Border ring rebuilt. Starts unchanged. Made by RS_수업자료/3주차/tools/make_teammap_long.py"
stage.GetDefaultPrim().SetCustomDataByKey("map_metadata", json.dumps(md))
stage.GetRootLayer().Export(DST)
sha = hashlib.sha256(open(DST, "rb").read()).hexdigest()
out_json = dict(md)
out_json["usd_file"] = os.path.basename(DST)
out_json["usd_sha256"] = sha
# 팀원 JSON 의 키 순서를 흉내 (usd_file, usd_sha256 는 맨 뒤)
json.dump(out_json, open(DST[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n".join(report[:3] + ["..."] + report[-2:]))
print(f"썼음: {DST} ({os.path.getsize(DST) / 1e6:.1f} MB), sha256 {sha[:16]}…, 행 {md['generator']['rows']}, 타일 {len(md['tiles'])}, 출발점 {len(md['evaluation']['starts'])}")
