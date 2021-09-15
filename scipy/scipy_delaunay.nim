import numericalnim
import plotly
import nimpy
import arraymancer
import arraymancer/tensor/private/p_accessors

import timelog
import ./npytensors
import ./gridutils

import std/browsers
import std/math
import std/os

# from scipy.interpolate import CloughTocher2DInterpolator
proc cloughTocher*[T](regrid_x, regrid_y, igrid, orig: Tensor[T]): Tensor[T] =
  let interp = pyImport("scipy.interpolate")
  let Nx = igrid.shape[0]
  let Ny = igrid.shape[1]

  var
    pyigrid = toNdArray[float](igrid.reshape((Nx*Ny).int, 2))
    pyorig = toNdArray[float](orig.reshape((Nx*Ny).int))

  var
    cloughTocherInterpolator = interp.CloughTocher2DInterpolator(pyigrid, pyorig, 0.0)
    pyinterpres = nimpy.callMethod(cloughTocherInterpolator, "__call__", toNdArray(regrid_x), toNdArray(regrid_y))
  result = toTensor[float](pyinterpres)


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

proc genContigRandDatas*[T](Nx, Ny: float): Tensor[T] =
  # Create datas
  var false_orig = genRand(toInt(Nx/10)+1, toInt(Ny/10)+1)
  var bicubic = newBicubicSpline(false_orig,
                                 (0.0, (Nx)),
                                 (0.0, (Ny))
                                )

  result = newTensor[T]((Nx).int, (Ny).int)
  for c, x in result.mpairs:
    if c[0] >= result.shape[0] or c[1] >= result.shape[1]:
      x = bicubic.eval(c[0].float-1, c[1].float-1)
    else:
      x = bicubic.eval(c[0].float, c[1].float)

proc sinc[T: SomeFloat](x: T): T =
  if x == T(0.0):
    T(1.0)
  else:
    sin(x)/x

proc fdata[T: SomeFloat](x: T, y: T): T =
  let c = pow(x*x+y*y, 0.5)
  # abs(sinc(c))
  pow(abs(sinc(c)), 0.5)

proc fcoord(x, y: int, Nx, Ny: int): tuple[x, y: float] =
  let x = x.float
  let y = y.float

  let
    x_0 = Nx/3
    y_0 = Ny/3

    x_2 = 2*Nx/3
    y_2 = 2*Ny/3

  let
    cx = if x <= x_0: (x-Nx/6) elif x >= x_2: (x-5*Nx/6) else: (x-Nx/2)
    cy = if y <= y_0: (y-Ny/6) elif y >= y_2: (y-5*Ny/6) else: (y-Ny/2)

  result.x = cx
  result.y = cy

proc genContigPeriodicDatas*(Nx, Ny: float): Tensor[float] =
  result = newTensor[float]((Nx).int, (Ny).int)
  for c, val in result.mpairs:
    let (cx, cy) = fcoord(c[0], c[1], Nx.int, Ny.int)
    val = fdata(cx, cy)

proc main() =
  let
    Nx = 200.0
    dx = 100e-6
    Ny = 240.0
    dy = 100e-6

  var
    igrid_x = irgrid_x(Nx, Ny, dx, dy)
    igrid_y = irgrid_y(Nx, Ny, dx, dy)
    regrid_x = rgrid_x(Nx, Ny, dx, dy)
    regrid_y = rgrid_y(Nx, Ny, dx, dy)

  var
    igrid = merge_grids(igrid_x, igrid_y)
    rgrid = merge_grids(regrid_x, regrid_y)
    igrids = igrid.grid_aspoints
    rgrids = rgrid.grid_aspoints

  # var datas = genContigRandDatas[float](Nx.float, Ny.float)
  var datas = genContigPeriodicDatas(Nx.float, Ny.float)

  let p0 = plotheatmap(datas, "Datas on regular grid")
  openDefaultBrowser(p0.save("/tmp/x0.html"))

  timeIt("precalc_delaunay"):
    var (triangleDist, simplices, triangleIdx) = precalc_delaunay(igrids, rgrids)

  timeIt("eval_delaunay"):
    var delaunay = eval_delaunay(datas, triangleDist, simplices, triangleIdx)

  block:
    let mre = mean_relative_error(datas, delaunay)
    echo "mean_relative_error=", mre

    let p1 = plotheatmap(delaunay, "Datas on irregular grid")
    openDefaultBrowser(p1.save("/tmp/x1.html"))

  timeIt("cloughTocher"):
    var clough_tocher_datas = cloughTocher(regrid_x, regrid_y, igrid, datas)

  block:
    let mre = mean_relative_error(datas, clough_tocher_datas)
    echo "mean_relative_error=", mre

    let p2 = plotheatmap(clough_tocher_datas, "Clough Tocher datas")
    openDefaultBrowser(p2.save("/tmp/x2.html"))

when isMainModule:
  main()
