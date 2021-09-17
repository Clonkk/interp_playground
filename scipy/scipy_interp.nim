import numericalnim
import plotly
import nimpy
import arraymancer
import arraymancer/tensor/private/p_accessors

import timelog
import ./npytensors
import ./gridutils
import ./scipy_spatial

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

proc interpgrid*[T](igrid, orig, rgrid: Tensor[T]): Tensor[T] =
  let interp = pyImport("scipy.interpolate")

  let Nx = igrid.shape[0]
  let Ny = igrid.shape[1]

  var
    pyigrid = toNdArray[float](igrid.reshape((Nx*Ny).int, 2))
    pyorig = toNdArray[float](orig.reshape((Nx*Ny).int))
    pyrgrid = toNdArray[float](rgrid.reshape((Nx*Ny).int, 2))

  var griddata = interp.griddata(pyigrid.PyObject, pyorig.PyObject, pyrgrid.PyObject, "linear", 0.0)
  result = toTensor[float](griddata).reshape(Nx, Ny)

proc eval_barycentric(points, values, xi: Tensor[float]) : Tensor[float]=
  let
    Nx = points.shape[0]
    Ny = points.shape[1]
    Npts = Nx*Ny

  var
    points = points.reshape(Npts, 2)
    values = values.reshape(NPts.int)
    # xi = xi.reshape(NPts, 2)

  timeIt("newBarycentric2D"):
    let bary = newBarycentric2D(points, values)

  # Remove border to avoid out of the convex hull element
  result = newTensor[float]([Nx-2, Ny-2])
  timeIt("evalBarycentric"):
    for coord, val in result.mpairs:
      let
        x = xi[coord[0]+1, coord[1]+1, 0]
        y = xi[coord[0]+1, coord[1]+1, 1]
      val = bary.eval(x, y)


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
  var regDatas = genContigPeriodicDatas(Nx.float, Ny.float)
  let p0 = plotheatmap(regDatas, "Regular Datas")
  openDefaultBrowser(p0.save("/tmp/x0.html"))

  # Deform data using an interpolation
  var datas = interpgrid(rgrid, regDatas, igrid)
  let p1 = plotheatmap(datas, "Datas")
  openDefaultBrowser(p1.save("/tmp/x1.html"))

  timeIt("griddata_interp"):
    var res_grid = interpgrid(igrid, datas, rgrid)

  block:
    let mre = mean_relative_error(res_grid, datas)
    echo "mean_relative_error=", mre
    let p2 = plotheatmap(res_grid, "Scipy Griddata")
    openDefaultBrowser(p2.save("/tmp/x2.html"))

  timeIt("barycentric_interp"):
    var bary_datas = eval_barycentric(igrid, datas, rgrid)
  var res = zeros_like(datas)
  res[1..^2, 1..^2] = bary_datas

  block nativeBarycentric:
    let mre = mean_relative_error(datas, res)
    echo "mean_relative_error=", mre
    let p3 = plotheatmap(bary_datas, "Barycentric datas")
    openDefaultBrowser(p3.save("/tmp/x2.html"))

  echo "mean_relative_error(griddata, barycenter)=", mean_relative_error(res, res_grid)

when isMainModule:
  main()
