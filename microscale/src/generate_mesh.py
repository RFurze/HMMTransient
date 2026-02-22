from netgen.csg import *
from math import pi
from netgen.meshing import *
from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor, Mesh
from netgen.csg import Pnt
import ngsolve



def GenerateSquareMesh_Scaled(N, Lx, Ly):
    N = int(N/2)
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
                mesh.Add(Element2D(1, [pnums[i + j * (N + 1)], pnums[i + (j + 1) * (N + 1)],
                                       pnums[i + 1 + (j + 1) * (N + 1)], pnums[i + 1 + j * (N + 1)]]))
            else:
                mesh.Add(
                    Element2D(1, [pnums[i + j * (N + 1)], pnums[i + (j + 1) * (N + 1)], pnums[i + 1 + j * (N + 1)]]))
                mesh.Add(Element2D(1, [pnums[i + (j + 1) * (N + 1)], pnums[i + 1 + (j + 1) * (N + 1)],
                                       pnums[i + 1 + j * (N + 1)]]))
    # Set horizontal boundary elements
    for i in range(N):
        mesh.Add(Element1D([pnums[N + i * (N + 1)], pnums[N + (i + 1) * (N + 1)]], index=1))  # add the top
        mesh.Add(Element1D([pnums[0 + i * (N + 1)], pnums[0 + (i + 1) * (N + 1)]], index=2))  # add the bottom
    # Set vertical boundary elements
    for i in range(N):
        mesh.Add(Element1D([pnums[i], pnums[i + 1]], index=3))  # add the left boundary
        mesh.Add(Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=4))  # add the right

    # Identify points for the x-direction
    for j in range(N + 1):
        for i in range(N + 1):
            if i == N:
                slave = slave_points_x[j]
                mesh.AddPointIdentification(pnums[i * (N + 1) + j], slave, identnr=1, type=2)
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)

    # Identify points for the y-direction
    for i in range(N + 1):
        for j in range(N + 1):
            if j == N:
                slave = slave_points_y[i]
                mesh.AddPointIdentification(pnums[i * (N + 1) + j], slave, identnr=2, type=2)
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)

    mesh.SetBCName(0, "top")
    mesh.SetBCName(1, "bottom")
    mesh.SetBCName(2, "left")
    mesh.SetBCName(3, "right")

    meshout = ngsolve.Mesh(mesh)
    boundaries = meshout.GetBoundaries()
    return meshout


if __name__ == "__main__":
    # Generate the mesh
    N = 100
    Lx = 250e-6
    Ly = 250e-6
    generated_mesh = GenerateSquareMesh_Scaled(N, Lx, Ly)
    # Save the mesh to a file
    generated_mesh.ngmesh.Save("N100_xy250um_mesh.vol")
    print("Mesh saved to my_mesh.vol")
