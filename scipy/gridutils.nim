import arraymancer
import plotly

import std/sequtils
import std/macros
import std/random

randomize()

type
  Point*[T] = tuple[x: T, y: T]
  Triangle*[T]= tuple[a, b, c: Point[T]]

proc tracetriangle*[T](triangle: Triangle[T], title: string): Trace[T] =
  result = Trace[float](mode: PlotMode.LinesMarkers, `type`: PlotType.Scatter, name: title)

  result.xs.add(triangle.a.x)
  result.ys.add(triangle.a.y)

  result.xs.add(triangle.b.x)
  result.ys.add(triangle.b.y)

  result.xs.add(triangle.c.x)
  result.ys.add(triangle.c.y)

  result.xs.add(triangle.a.x)
  result.ys.add(triangle.a.y)

proc tracegrid*[T](posxy: Tensor[Point[T]], title: string): Trace[T] =
  let
    d = Trace[T](mode: PlotMode.Markers, `type`: PlotType.Scatter, name: title)
  var posx: Tensor[T] = posxy.map_inline:
    x.x

  var posy: Tensor[T] = posxy.map_inline:
    x.y

  d.xs = toSeq(posx)
  d.ys = toSeq(posy)
  return d

proc tracegrid*[T](posx: Tensor[T], posy: Tensor[T]): Trace[T] =
  let
    d = Trace[T](mode: PlotMode.Markers, `type`: PlotType.Scatter, name: title)

  d.xs = toSeq(posx)
  d.ys = toSeq(posy)
  return d

proc traceheatmap*[T](t: Tensor[T]): Trace[T] =
  let
    # The GL heatmap is also supported as HeatMapGL
    d = Trace[T](mode: PlotMode.Lines, `type`: PlotType.HeatMap)
  d.colormap = ColorMap.Viridis
# fill data for colormap with random values. The data needs to be supplied
# as a nested seq.
  d.zs = t.toSeq2D
  return d

proc plotheatmap*[T](t: Tensor[T], title: string = ""): Plot[T] =
  let
    d = traceheatmap(t)
    layout = Layout(title: title, width: 1400, height: 1000,
                    xaxis: Axis(title: "x-"&title),
                    yaxis: Axis(title: "y-"&title),
                    autosize: false)
    p = Plot[T](layout: layout, traces: @[d])
  return p


proc plotTraces*[T](d: seq[Trace[T]], title: string): Plot[T] =
  let
    layout = Layout(title: title, width: 1600, height: 1000,
                    xaxis: Axis(title: "x-axis"),
                    yaxis: Axis(title: "y-axis"),
                    autosize: false)

    p = Plot[T](layout: layout, traces: d)
  return p

template subplotTraces*[T](h1, h2: Trace[T], title: string): Plot[T] =

  let baseLayout = Layout(title: title, width: 1600, height: 1000,
                          xaxis: Axis(title: "x"),
                          yaxis: Axis(title: "y"),
                          autosize: false)

  let p = subplots:
    baseLayout: baseLayout
    plot: h1
    plot: h2

  return p


proc rgrid_x*(Nx, Ny, dx, dy: float): Tensor[float]  =
  result = newTensor[float](Nx.int, Ny.int)
  for coord, val in result.mpairs:
    let x = coord[0].float
    val = x*dx

proc rgrid_y*(Nx, Ny, dx, dy: float): Tensor[float]  =
  result = newTensor[float](Nx.int, Ny.int)
  for coord, val in result.mpairs:
    let y = coord[1].float
    val = y*dy


proc genRand*(Nx, Ny: int): Tensor[float]  =
  result = randomTensor([Nx, Ny], 80.0).asType(float)

proc genRand*(Nx, Ny, Nt: int): Tensor[float] =
  result = randomTensor([Nx, Ny, Nt], 80.0).asType(float)

const err_small_coeff = 0.3
const err_large_coeff = 2.1

proc irgrid_x*(Nx, Ny, dx, dy: float): Tensor[float]  =
  result = newTensor[float](Nx.int, Ny.int)
  for coord, val in result.mpairs:
    var r = rand(100)
    var err = rand(2*err_small_coeff)-err_small_coeff
    if r > 90:
      err = rand(2*err_large_coeff)-err_large_coeff

    let x = coord[0].float
    val = (x+err)*dx
    while val < 0:
      val += rand(0.2)*dx
    while val > dx*(Nx-1):
      val -= rand(0.2)*dx

proc irgrid_y*(Nx, Ny, dx, dy: float): Tensor[float]  =
  result = newTensor[float](Nx.int, Ny.int)
  for coord, val in result.mpairs:
    var r = rand(100)
    var err = rand(2*err_small_coeff)-err_small_coeff
    if r > 90:
      err = rand(2*err_large_coeff)-err_large_coeff

    let y = coord[1].float
    val = (y+err)*dy
    while val < 0:
      val += rand(0.2)*dy
    while val > dy*(Ny-1):
      val -= rand(0.2)*dy

proc select_subtensor*[T](igrid: Tensor[T], xbounds: tuple[lower: int, upper: int], ybounds: tuple[lower: int,
    upper: int]): Tensor[T] =
  let
    minx = xbounds.lower
    maxx = xbounds.upper
    miny = ybounds.lower
    maxy = ybounds.upper

  result = newTensor[T](maxx-minx, maxy-miny, 2)
  for c, g in result.mpairs:
    let
      i = c[0]
      j = c[1]
      k = c[2]
    g = igrid[i, j, k]

proc grid_astensor*[T](grid: Tensor[Point[T]]): Tensor[T] =
  result = newTensor[T](grid.shape[0], grid.shape[1], 2)
  for c, g in result.mpairs:
    let i = c[0]
    let j = c[1]
    let k = c[2]
    case k
    of 0:
      g = grid[i, j].x
    of 1:
      g = grid[i, j].y
    else:
      discard

proc grid_aspoints*[T](grid: Tensor[T]): Tensor[Point[T]] =
  result = newTensor[Point[T]](grid.shape[0], grid.shape[1])
  for c, g in result.mpairs:
    let i = c[0]
    let j = c[1]
    let tmp = Point[T]((x: grid[i, j, 0], y: grid[i, j, 1]))
    g = tmp

proc merge_grids*[T](grid_x, grid_y: Tensor[T]) : Tensor[T] =
  assert grid_x.shape == grid_y.shape
  result = newTensor[float](grid_x.shape[0], grid_x.shape[1], 2)
  for coord, x in result.mpairs:
    case coord[2]
    of 0:
      x = grid_x[coord[0], coord[1]]
    of 1:
      x = grid_y[coord[0], coord[1]]
    else:
      discard
