import numericalnim
import arraymancer

import std/sequtils
import std/browsers
import std/random
import std/sugar
import std/math
import std/os

randomize()

# Create an irregular points based on a random factor
proc randPoint*(x: int, min_factor, max_factor: float, xlim: (float, float), dX: float): float =
  result = (x.float + rand(min_factor..max_factor))*dX
  if result < xlim[0]:
    result = xlim[0]
  elif result > xlim[1]:
    result = xlim[1]

# Create a vector of irregular coordinate
proc IRGridVec*(nx: int, xlim: (float, float)): seq[float] =
  let
    randfactor = 0.2
    dX = (xlim[1]-xlim[0])/nx.float
  result = toSeq(0..<nx).map(x => randPoint(x, -randfactor, randfactor, xlim, dX))
  # echo "IRGridVec> ", result.len

# Create a vector of regular coordinate
proc RGridVec*(nx: int, xlim: (float, float)): seq[float] =
  let dX = (xlim[1]-xlim[0])/nx.float
  result = toSeq(0..<nx).map(x => (x.float)*dx)
  # echo "RGridVec> ", result.len

# Convert 2 vector of coordinate into a grid
proc gridsToPoints*[T](gridx, gridy: seq[T]): Tensor[Complex[T]] =
  result = newTensor[Complex[T]](gridx.len, gridy.len)
  # echo ">> ", result.shape
  for c, g in result.mpairs:
    let i = c[0]
    let j = c[1]
    g.re = gridx[i]
    g.im = gridy[j]
  # echo "grids> ", result.shape

# Use a bicubic interpolation to generate a Tensor of datas without too much gradient
proc randGradDatas(N: (int, int), f: int, xlim, ylim: (float, float)): tuple[pX, pY: seq[float], datas: Tensor[float]] =
  let
    nx = N[0]
    ny = N[1]

  let
    origDatas = randomTensor[float]([nx, ny], 10.0)

  result.pX = RGridVec(origDatas.shape[0]*f, xlim)
  result.pY = RGridVec(origdatas.shape[1]*f, ylim)

  var interp = newBicubicSpline(origDatas, xlim, ylim)
  result.datas = newTensor[float](nx*f, ny*f)
  for (c, e) in result.datas.mpairs:
    let
      x = result.pX[c[0]]
      y = result.pY[c[1]]

    e = interp.eval(x, y)

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

proc periodicDatas(N: (int, int), f: int, xlim, ylim: (float, float)): tuple[pX, pY: seq[float], datas: Tensor[float]] =
  let
    nx = N[0]*f
    ny = N[1]*f

  result.pX = RGridVec(nx, xlim)
  result.pY = RGridVec(ny, ylim)
  var datas = newTensor[float]((nx).int, (ny).int)

  for c, val in datas.mpairs:
    let (cx, cy) = fcoord(c[0], c[1], nx.int, ny.int)
    val = fdata(cx, cy)

  result.datas = datas

######################
# Plotly utility procs
######################
import plotly
proc traceheatmap[T](t: Tensor[T]): Trace[T] =
  let
    # The GL heatmap is also supported as HeatMapGL
    d = Trace[T](mode: PlotMode.Lines, `type`: PlotType.HeatMap)
  d.colormap = ColorMap.Viridis
# fill data for colormap with random values. The data needs to be supplied
# as a nested seq.
  d.zs = t.toSeq2D
  return d

proc plotheatmap*[T](t: Tensor[T], title: string): Plot[T] =
  let
    d = traceheatmap(t)
    layout = Layout(title: title, width: 1400, height: 1000,
                    xaxis: Axis(title: "x-"&title),
                    yaxis: Axis(title: "y-"&title),
                    autosize: false)
    p = Plot[T](layout: layout, traces: @[d])
  return p

proc tracegrid*[T](pos: Tensor[Complex[T]], title: string = ""): Trace[T] =
  let
    d = Trace[T](mode: PlotMode.Markers, `type`: PlotType.Scatter, name: title)

  for p in pos:
    d.xs.add p.re
    d.ys.add p.im
  return d

proc tracegrid*[T](posx, posy: seq[T], title: string = ""): Trace[T] =
  let pos = gridsToPoints(posx, posy)
  return tracegrid(pos, title)

proc plotTraces*[T](d: seq[Trace[T]], title = ""): Plot[T] =
  let
    layout = Layout(title: "", width: 1600, height: 1000,
                    xaxis: Axis(title: "x-axis"),
                    yaxis: Axis(title: "y-axis"),
                    autosize: false)

    p = Plot[T](layout: layout, traces: d)
  return p
################################
# End of plotly specific section
################################

proc irregularInterp[T](datas: Tensor[T], X, Y: (float, float), pX, pY: seq[float]) : Tensor[T] =
  # Interpolator on the regular grid of our initial datas
  var interp = newBicubicSpline(datas, X, Y)
  result = newTensor[T](datas.shape.toSeq)
  # result = zeros_like(datas)

  echo "Interpolate..."
  # Calculate the data on the irregular grid
  for idx, e in mpairs(result):
    var
      x = pX[idx[0]]
      y = pY[idx[1]]
    e = interp.eval(x, y)

proc nimInterp() =
  let
    orignx = 6
    origny = 4
    dimension_multipication_factor  = 10
    nx = dimension_multipication_factor*orignx
    ny = dimension_multipication_factor*origny
    X = (0.0, ny.float) # This force pixel size to 1 so we can display pretty heatmap with the grids on top of it
    Y = (0.0, nx.float) # I'm just too lazy to change the voxel size of the heatmap, it's just easier to do it this way

  # Here we do the opposite of what we want: we have datas on a regular grid and we can use numericalnim to calculate data on the irregular grid
  # var (prX, prY, datas) = randGradDatas(N = (orignx, origny), f = dimension_multipication_factor, xlim = X, ylim = Y)
  var (prX, prY, datas) = periodicDatas(N = (orignx, origny), f = dimension_multipication_factor, xlim = X, ylim = Y)
  # echo datas.shape

  # Create irregular grid
  var pX = IRGridVec(datas.shape[0], X)
  var pY = IRGridVec(datas.shape[1], Y)

  var res = irregularInterp(datas, X, Y, pX, pY)

  # Plotting result
  echo "Plotting..."

  block: # Trace grid only
    let traces = @[tracegrid(prX, prY), tracegrid(pX, pY)]
    var p = plotTraces(traces, "Numerical Nim - Grid only")
    let savepath = p.save("/tmp/x0.html")
    openDefaultBrowser(savepath)

  block: # Trace grid + original data
    let traces = @[tracegrid(prX, prY), tracegrid(pX, pY), traceheatmap(datas)]
    var p = plotTraces(traces, "Numerical Nim - Orig Data")
    let savepath = p.save("/tmp/x1.html")
    openDefaultBrowser(savepath)


  block: # Trace grid + interpolated data
    let traces = @[tracegrid(prX, prY), tracegrid(pX, pY), traceheatmap(res)]
    var p = plotTraces(traces, "Numerical Nim - Interp Data")
    let savepath = p.save("/tmp/x2.html")
    openDefaultBrowser(savepath)

when isMainModule:
  nimInterp()
