from netgen.csg import *
from math import pi, sin
from netgen.meshing import *
import ngsolve


def GenerateSquareMesh_Scaled(N, Lx, Ly):
    N = int(N / 2)
    quads = True
    mesh = Mesh()
    mesh.SetGeometry(unit_square)
    mesh.dim = 2
    pnums = []
    coordinates = []  # To store the coordinates of the points
    slave_points_x = {}
    slave_points_y = {}
    matched_points = set()  # To store points that have been matched

    for i in range(N + 1):
        for j in range(N + 1):
            # Scale the coordinates
            x = (i / N) * Lx
            y = (j / N) * Ly
            point = Pnt(x, y, 0)
            pnums.append(mesh.Add(MeshPoint(point)))
            coordinates.append(point)

            if i == 0:
                # Store the first point in each column as a slave for x-direction
                slave_points_x[j] = pnums[-1]

            if j == 0:
                # Store the first point in each row as a slave for y-direction
                slave_points_y[i] = pnums[-1]

    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))

    mesh.SetMaterial(1, "mat")
    for j in range(N):
        for i in range(N):
            if quads:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
            else:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
    # Set horizontal boundary elements
    for i in range(N):
        mesh.Add(
            Element1D([pnums[N + i * (N + 1)], pnums[N + (i + 1) * (N + 1)]], index=1)
        )  # add the top
        mesh.Add(
            Element1D([pnums[0 + i * (N + 1)], pnums[0 + (i + 1) * (N + 1)]], index=2)
        )  # add the bottom
    # Set vertical boundary elements
    for i in range(N):
        mesh.Add(Element1D([pnums[i], pnums[i + 1]], index=3))  # add the left boundary
        mesh.Add(
            Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=4)
        )  # add the right

    # Identify points for the x-direction
    for j in range(N + 1):
        for i in range(N + 1):
            if i == N:
                slave = slave_points_x[j]
                mesh.AddPointIdentification(
                    pnums[i * (N + 1) + j], slave, identnr=1, type=2
                )
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)

    # Identify points for the y-direction
    for i in range(N + 1):
        for j in range(N + 1):
            if j == N:
                slave = slave_points_y[i]
                mesh.AddPointIdentification(
                    pnums[i * (N + 1) + j], slave, identnr=2, type=2
                )
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)

    mesh.SetBCName(0, "top")
    mesh.SetBCName(1, "bottom")
    mesh.SetBCName(2, "left")
    mesh.SetBCName(3, "right")

    meshout = ngsolve.Mesh(mesh)
    boundaries = meshout.GetBoundaries()
    return meshout


def GenerateSquareMesh(N, Lx, Ly):
    N = int(N / 2)
    quads = True
    mesh = Mesh()
    mesh.SetGeometry(unit_square)
    mesh.dim = 2
    pnums = []
    coordinates = []  # To store the coordinates of the points
    slave_points_x = {}
    slave_points_y = {}
    matched_points = set()  # To store points that have been matched

    for i in range(N + 1):
        for j in range(N + 1):
            point = Pnt(i / N, j / N, 0)
            pnums.append(mesh.Add(MeshPoint(point)))
            coordinates.append(point)

            if i == 0:
                # Store the first point in each column as a slave for x-direction
                slave_points_x[j] = pnums[-1]

            if j == 0:
                # Store the first point in each column as a slave for y-direction
                slave_points_y[i] = pnums[-1]

    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))

    mesh.SetMaterial(1, "mat")
    for j in range(N):
        for i in range(N):
            if quads:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
            else:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
    # Set horizontal boundary elements
    for i in range(N):
        mesh.Add(
            Element1D([pnums[N + i * (N + 1)], pnums[N + (i + 1) * (N + 1)]], index=1)
        )  # add  the top
        mesh.Add(
            Element1D([pnums[0 + i * (N + 1)], pnums[0 + (i + 1) * (N + 1)]], index=2)
        )  # add the bottom
    # Set vertical boundary elements
    for i in range(N):
        mesh.Add(Element1D([pnums[i], pnums[i + 1]], index=3))  # add the left boundary
        mesh.Add(
            Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=4)
        )  # add the right

    # Identify points for the x-direction
    for j in range(N + 1):
        for i in range(N + 1):
            if i == N:
                identified_coords = coordinates[i * (N + 1) + j]
                slave = slave_points_x[j]
                slave_coords = coordinates[
                    pnums.index(slave)
                ]  # Get the coordinates of the slave point
                mesh.AddPointIdentification(
                    pnums[i * (N + 1) + j], slave, identnr=1, type=2
                )
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)
                # print(f'Identified point at {identified_coords} with slave point at {slave_coords} for x direction, difference = ({identified_coords[0]-slave_coords[0]}, {identified_coords[1]-slave_coords[1]}, {identified_coords[2]-slave_coords[2]})')

    # Identify points for the y-direction
    for i in range(N + 1):
        for j in range(N + 1):
            if j == N:
                identified_coords = coordinates[i * (N + 1) + j]
                slave = slave_points_y[i]
                slave_coords = coordinates[
                    pnums.index(slave)
                ]  # Get the coordinates of the slave point
                mesh.AddPointIdentification(
                    pnums[i * (N + 1) + j], slave, identnr=2, type=2
                )
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)
                # print(f'Identified point at {identified_coords} with slave point at {slave_coords} for y direction, difference = ({identified_coords[0]-slave_coords[0]}, {identified_coords[1]-slave_coords[1]}, {identified_coords[2]-slave_coords[2]})')

    mesh.SetBCName(0, "top")
    mesh.SetBCName(1, "bottom")
    mesh.SetBCName(2, "left")
    mesh.SetBCName(3, "right")

    meshout = ngsolve.Mesh(mesh)
    boundaries = meshout.GetBoundaries()
    print("Boundaries:", boundaries)
    return meshout


from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import (
    Element0D,
    Element1D,
    Element2D,
    MeshPoint,
    FaceDescriptor,
    Mesh,
)
from netgen.csg import Pnt


def GenerateSquareMeshBasic(N, Lx, Ly):
    quads = True
    mesh = Mesh()
    mesh.SetGeometry(unit_square)
    mesh.dim = 2
    pnums = []
    for i in range(N + 1):
        for j in range(N + 1):
            pnums.append(mesh.Add(MeshPoint(Pnt(i / N, j / N, 0))))

    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))

    mesh.SetMaterial(1, "mat")
    for j in range(N):
        for i in range(N):
            if quads:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
            else:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )

    for i in range(N):
        mesh.Add(
            Element1D([pnums[N + i * (N + 1)], pnums[N + (i + 1) * (N + 1)]], index=1)
        )
        mesh.Add(
            Element1D([pnums[0 + i * (N + 1)], pnums[0 + (i + 1) * (N + 1)]], index=2)
        )

    for i in range(N):
        mesh.Add(Element1D([pnums[i], pnums[i + 1]], index=3))
        mesh.Add(
            Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=4)
        )

    mesh.SetBCName(0, "top")
    mesh.SetBCName(1, "bottom")
    mesh.SetBCName(2, "left")
    mesh.SetBCName(3, "right")
    meshout = ngsolve.Mesh(mesh)

    return meshout


from netgen.meshing import *
import ngsolve
import math

import math
from netgen.meshing import *
import ngsolve


def GenerateSquareMesh_Scaled_Adaptive(
    N,
    Lx,
    Ly,
    H0=0.5e-8,
    h_fraction=0.2,  # refine cells with avg(h) < h_fraction * global_max
    max_refine=3,  # maximum refinement level
    quads=False,
):
    """
    Creates a 2D periodic domain [0,Lx] x [0,Ly] with multi-level refinement
    based solely on small average thickness h.

    Modifications:
      1) Boundary cells are limited to refine_level <= 1 (avoids super-fine boundary cells).
      2) Skip zero-length boundary segments.

    Steps:
      1) Build Nx x Nx grid of 'CellData' with refine_level=0.
      2) Cells with avg h < (h_fraction * global_max) => refine_level = max_refine
      3) Impose smoothness: neighbor refine levels differ by <= 1
      4) Build final mesh by subdividing each cell 2^(refine_level) times
      5) Add boundary segments, skip degenerate segments
      6) Apply periodic identifications
      7) Return as ngsolve.Mesh
    """

    sin = math.sin
    pi = math.pi

    # ---------------------------------------------
    # 1) Dimensionless thickness function
    # ---------------------------------------------
    def hhat(x, y):
        return (H0 + 0.5 * (1e-8) * sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly)) / H0

    dx = Lx / N
    dy = Ly / N

    # Sample the domain to find global min, max
    vals = []
    for i in range(N + 1):
        for j in range(N + 1):
            xx = i * dx
            yy = j * dy
            vals.append(hhat(xx, yy))
    global_min = min(vals)
    global_max = max(vals)
    if abs(global_max - global_min) < 1e-20:
        global_max = 1.0  # avoid div-by-zero

    threshold_val = h_fraction * global_max

    print("-----------------------------------------------------------")
    print(f"[Multi-Level] Domain = {Lx} x {Ly},   N = {N},   max_refine = {max_refine}")
    print(
        f"[Multi-Level] Global thickness range = [{global_min:.4g}, {global_max:.4g}]"
    )
    print(
        f"[Multi-Level] h_threshold = {threshold_val:.4g}  (cells below this => refine_level={max_refine})"
    )
    print("-----------------------------------------------------------")

    # ----------------------------------------------------
    # 2) Build Nx x Nx array of cells
    # ----------------------------------------------------
    class CellData:
        __slots__ = ("i", "j", "x0", "x1", "y0", "y1", "refine_level", "avg_h")

        def __init__(self, i, j, x0, x1, y0, y1):
            self.i = i
            self.j = j
            self.x0, self.x1 = x0, x1
            self.y0, self.y1 = y0, y1
            self.refine_level = 0  # start at 0
            self.avg_h = 0.0  # will fill later

    # Create a 2D list of cells
    cells_2d = []
    for i in range(N):
        row = []
        for j in range(N):
            x0 = i * dx
            x1 = (i + 1) * dx
            y0 = j * dy
            y1 = (j + 1) * dy
            row.append(CellData(i, j, x0, x1, y0, y1))
        cells_2d.append(row)

    # ----------------------------------------
    # 3) Compute average h for each cell
    # ----------------------------------------
    for i in range(N):
        for j in range(N):
            c = cells_2d[i][j]
            corners = [
                hhat(c.x0, c.y0),
                hhat(c.x1, c.y0),
                hhat(c.x0, c.y1),
                hhat(c.x1, c.y1),
            ]
            c.avg_h = sum(corners) / 4.0

    # ----------------------------------------------------------------
    # 4) Mark cells refine_level=max_refine if avg_h < threshold_val
    # ----------------------------------------------------------------
    for i in range(N):
        for j in range(N):
            c = cells_2d[i][j]
            if c.avg_h < threshold_val:
                c.refine_level = max_refine

    # ----------------------------------------------------------------
    # (a) Limit boundary cells to refine_level <= 1
    # ----------------------------------------------------------------
    for i in range(N):
        for j in range(N):
            c = cells_2d[i][j]
            # If cell is on boundary (top/bottom/left/right),
            # restrict refine_level to <= 1
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                c.refine_level = min(c.refine_level, 1)

    # ----------------------------------------------------------------
    # 5) Smoothness pass: |Lc - Ln| <= 1 for neighbors
    # ----------------------------------------------------------------
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(N):
                c = cells_2d[i][j]
                Lc = c.refine_level

                neighbors = []
                if i > 0:
                    neighbors.append(cells_2d[i - 1][j])
                if i < N - 1:
                    neighbors.append(cells_2d[i + 1][j])
                if j > 0:
                    neighbors.append(cells_2d[i][j - 1])
                if j < N - 1:
                    neighbors.append(cells_2d[i][j + 1])

                for nb in neighbors:
                    Ln = nb.refine_level
                    # If Ln > Lc+1 => raise c
                    if Ln > Lc + 1:
                        newL = min(Ln - 1, max_refine)
                        if newL > Lc:
                            c.refine_level = newL
                            changed = True
                    # If Lc > Ln+1 => raise nb
                    if Lc > Ln + 1:
                        newL = min(Lc - 1, max_refine)
                        if newL > Ln:
                            nb.refine_level = newL
                            changed = True

    # Print distribution of refine levels for debugging
    level_counts = {}
    for i in range(N):
        for j in range(N):
            L = cells_2d[i][j].refine_level
            level_counts[L] = level_counts.get(L, 0) + 1
    print("Refine Level Distribution after smoothing:")
    for lev in sorted(level_counts.keys()):
        print(f"  Level = {lev},  #cells = {level_counts[lev]}")
    print("-----------------------------------------------------------")

    # -------------------------------------------------------------------------
    # 6) Build final sub-cells: each cell => 2^(refine_level) x 2^(refine_level)
    # -------------------------------------------------------------------------
    def subdivide_cell(x0, x1, y0, y1, nx, ny):
        subcells = []
        dx_sub = (x1 - x0) / nx
        dy_sub = (y1 - y0) / ny
        for ix in range(nx):
            for iy in range(ny):
                sx0 = x0 + ix * dx_sub
                sx1 = x0 + (ix + 1) * dx_sub
                sy0 = y0 + iy * dy_sub
                sy1 = y0 + (iy + 1) * dy_sub
                subcells.append((sx0, sx1, sy0, sy1))
        return subcells

    refined_quads = []
    for i in range(N):
        for j in range(N):
            c = cells_2d[i][j]
            level = c.refine_level
            n_sub = 2**level
            subs = subdivide_cell(c.x0, c.x1, c.y0, c.y1, n_sub, n_sub)
            refined_quads += subs

    total_subcells = len(refined_quads)
    print(f"Total sub-cells (2D elements) to be generated = {total_subcells}")
    print("-----------------------------------------------------------")

    # -------------------------------------------------------------
    # 7) Build Netgen Mesh
    # -------------------------------------------------------------
    mesh = Mesh()
    mesh.dim = 2
    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    mesh.SetMaterial(1, "mat")

    # Increase tolerance to reduce floating-point mismatch
    tol = 1e-14

    point_dict = {}

    def AddMeshPoint(x, y):
        # Enforce a bigger tolerance near Lx, Ly
        if abs(x - Lx) < tol:
            x = Lx
        if abs(y - Ly) < tol:
            y = Ly
        if (x, y) not in point_dict:
            mp = mesh.Add(MeshPoint(Pnt(x, y, 0)))
            point_dict[(x, y)] = mp
        return point_dict[(x, y)]

    # Add 2D elements
    cnt = 0
    for x0, x1, y0, y1 in refined_quads:
        p1 = AddMeshPoint(x0, y0)
        p2 = AddMeshPoint(x0, y1)
        p3 = AddMeshPoint(x1, y1)
        p4 = AddMeshPoint(x1, y0)

        if quads:
            mesh.Add(Element2D(1, [p1, p2, p3, p4]))
        else:
            # Triangulate each sub-quad
            mesh.Add(Element2D(1, [p1, p2, p4]))
            mesh.Add(Element2D(1, [p2, p3, p4]))

        cnt += 1

    unique_points = len(point_dict)
    print(f"Unique mesh points created: {unique_points}")
    print("-----------------------------------------------------------")

    # -------------------------------------------------------------
    # 8) Build boundary segments, skipping degenerate ones
    # -------------------------------------------------------------
    left_pts, right_pts = [], []
    bottom_pts, top_pts = [], []
    for (x, y), pnum in point_dict.items():
        if abs(x) < tol:
            left_pts.append((y, pnum))
        if abs(x - Lx) < tol:
            right_pts.append((y, pnum))
        if abs(y) < tol:
            bottom_pts.append((x, pnum))
        if abs(y - Ly) < tol:
            top_pts.append((x, pnum))

    left_pts.sort(key=lambda v: v[0])
    right_pts.sort(key=lambda v: v[0])
    bottom_pts.sort(key=lambda v: v[0])
    top_pts.sort(key=lambda v: v[0])

    def AddBoundarySegments(pts, idx):
        seg_count = 0
        for i in range(len(pts) - 1):
            pA = pts[i][1]
            pB = pts[i + 1][1]
            if pA != pB:  # skip if degenerate
                mesh.Add(Element1D([pA, pB], index=idx))
                seg_count += 1
        print(f"   boundary idx={idx}, #1D segments = {seg_count}")

    print("Boundary segments added:")
    AddBoundarySegments(top_pts, 1)  # top
    AddBoundarySegments(bottom_pts, 2)  # bottom
    AddBoundarySegments(left_pts, 3)  # left
    AddBoundarySegments(right_pts, 4)  # right
    print("-----------------------------------------------------------")

    # -------------------------------------------------------------
    # 9) Periodic boundary identification
    # -------------------------------------------------------------
    slave_x = {}
    slave_y = {}

    for (x, y), pnum in point_dict.items():
        # left edge
        if abs(x) < tol:
            slave_x[round(y, 8)] = pnum
        # bottom edge
        if abs(y) < tol:
            slave_y[round(x, 8)] = pnum

    # Identify x=0 with x=Lx
    for (x, y), pnum in point_dict.items():
        if abs(x - Lx) < tol:
            ky = round(y, 8)
            if ky in slave_x:
                s = slave_x[ky]
                if s != pnum:
                    mesh.AddPointIdentification(pnum, s, identnr=1, type=2)

    # Identify y=0 with y=Ly
    for (x, y), pnum in point_dict.items():
        if abs(y - Ly) < tol:
            kx = round(x, 8)
            if kx in slave_y:
                s = slave_y[kx]
                if s != pnum:
                    mesh.AddPointIdentification(pnum, s, identnr=2, type=2)

    # -------------------------------------------------------------
    # 10) Name boundaries and finalize
    # -------------------------------------------------------------
    mesh.SetBCName(0, "top")
    mesh.SetBCName(1, "bottom")
    mesh.SetBCName(2, "left")
    mesh.SetBCName(3, "right")

    print("-> Attempting to convert to NGSolve mesh now.")
    mesh.Save("debug_boundary.vol")
    ngmesh = ngsolve.Mesh(mesh)
    boundaries = ngmesh.GetBoundaries()
    print("Boundaries:", boundaries)
    print("-> Conversion to NGSolve mesh succeeded.")
    print("-----------------------------------------------------------")
    return ngmesh
