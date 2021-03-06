import nimpy
import arraymancer
import arraymancer/tensor/private/p_accessors

import ./npytensors
import ./gridutils

import std/math
import std/os


proc spatialDelaunay*[T](points: Tensor[Point[T]]): PyObject =
  let spatial = pyImport("scipy.spatial")
  let size = points.size.int

  # Reshape Tensor[Point] (Nx, Ny) => to Tensor[float] (Nx*Ny, 2) => to PyObject
  var points: Tensor[float] = points.grid_astensor.reshape(size, 2)
  var pyPts = toNdArray(points)
  result = spatial.Delaunay(pyPts)

proc triangulate_simplices*[T](simplices: Tensor[int32], igrids: Tensor[Point[T]]): Tensor[Triangle[T]] =

  var triangles: seq[Triangle[T]]
  for coord in simplices.axis(0):
    let elem = coord.squeeze
    var tri: Triangle[T]
    tri.a = igrids.atContiguousIndex(elem[0])
    tri.b = igrids.atContiguousIndex(elem[1])
    tri.c = igrids.atContiguousIndex(elem[2])
    triangles.add(tri)
  result = triangles.toTensor

proc precalc_delaunay*[T](igrids, rgrids: Tensor[T]): tuple[triangleDist: Tensor[float32], simplices: Tensor[int32], triangleIdx: Tensor[int32]] =
  var
    # PyObject using Qhull library
    pytri = spatialDelaunay(igrids)

  var
    # [Nx*Ny, 3] => contiguous index of tri-points forming delaunay triangles
    simplices = toTensor[int32](pytri.simplices)

  var
    # [Nx*Ny] => Triangles coordinate
    triangles = triangulate_simplices(simplices, igrids)

  var
    # [Nx, Ny] => contiguous index of triangles at [i, j]
    triangleIdx = newTensor[int32](rgrids.shape)

  for elem, value in mzip(triangleIdx, rgrids):
    let tri = pytri.find_simplex(value)
    # find triangle closest to regular grid
    elem = tri.to(int32)

  # [Nx, Ny, 3] => distance
  var triangleDist = newTensor[float32](rgrids.shape[0], rgrids.shape[1], 3)
  for coords, elem in triangleDist.mpairs:
    let
      coord = [coords[0], coords[1]]
      tri = triangleIdx.atIndex(coord)
      value = rgrids.atIndex(coord)

    if tri != -1:
      let triangle = triangles.atContiguousIndex(tri)
      # If regular grids associated points exists (tri != -1), calculate distances
      case coords[2]
      of 0:
        elem = 1 - sqrt(pow(triangle.a.x - value.x, 2.0) + pow(triangle.a.y - value.y, 2.0))
      of 1:
        elem = 1 - sqrt(pow(triangle.b.x - value.x, 2.0) + pow(triangle.b.y - value.y, 2.0))
      of 2:
        elem = 1 - sqrt(pow(triangle.c.x - value.x, 2.0) + pow(triangle.c.y - value.y, 2.0))
      else:
        discard

      # Sanity check
      assert elem >= 0 and elem <= 1

  # echo "Triangles.shape = ", triangles.shape
  # echo "Simplices.shape = ", simplices.shape
  # echo "triangleIdx.shape = ", triangleIdx.shape
  # echo "triangleDist.shape = ", triangleDist.shape

  result.triangleDist = triangleDist
  result.simplices = simplices
  result.triangleIdx = triangleIdx

proc eval_delaunay*[T](orig: Tensor[T], triangleDist: Tensor[float32], simplices: Tensor[int32], triangleIdx: Tensor[int32]): Tensor[T] =
  result = newTensor[float](orig.shape[0], orig.shape[1])

  for coord, value in result.mpairs:
    let
      dist = triangleDist[coord[0], coord[1], _].squeeze()
      tri_idx = triangleIdx.atIndex(coord)

    if tri_idx == -1:
      continue

    let simplice = simplices.atAxisIndex(0, tri_idx).squeeze()
    for j, d in dist:
      let contig_idx = simplice[j][0]
      value += d*orig.atContiguousIndex(contig_idx)
    value /= sum(dist)

