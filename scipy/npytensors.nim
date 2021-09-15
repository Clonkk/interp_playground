import std/sequtils
import std/strformat
import std/tables

import arraymancer

import nimpy
import nimpy/raw_buffers

proc dtype*(t: PyObject): PyObject =
  nimpy.getAttr(t, "dtype")

proc pyprint*(o: PyObject) =
  let py = pyBuiltinsModule()
  discard nimpy.callMethod(py, "print", o)

proc nptypes(name: string): PyObject =
  let np = pyImport("numpy")
  nimpy.getAttr(np, name)

template dtype*(T: typedesc[int8]): PyObject = nptypes("byte")
template dtype*(T: typedesc[int16]): PyObject = nptypes("short")
template dtype*(T: typedesc[int32]): PyObject = nptypes("intc")
template dtype*(T: typedesc[int64]): PyObject = nptypes("int")

template dtype*(T: typedesc[uint8]): PyObject = nptypes("ubyte")
template dtype*(T: typedesc[uint16]): PyObject = nptypes("ushort")
template dtype*(T: typedesc[uint32]): PyObject = nptypes("uintc")
template dtype*(T: typedesc[uint64]): PyObject = nptypes("uint")

proc dtype*(T: typedesc[int]): PyObject =
  when sizeof(T) == sizeof(int64):
    dtype(int64)
  elif sizeof(T) == sizeof(int32):
    dtype(int32)
  else:
    {.error: "Unsupported sizeof(uint)".}

proc dtype*(T: typedesc[uint]): PyObject =
  when sizeof(T) == sizeof(uint64):
    dtype(uint64)
  elif sizeof(T) == sizeof(uint32):
    dtype(uint32)
  else:
    {.error: "Unsupported sizeof(uint)".}

proc dtype*(T: typedesc[bool]): PyObject = nptypes("bool")
proc dtype*(T: typedesc[char]): PyObject = nptypes("char")
proc dtype*(T: typedesc[float32]): PyObject = nptypes("single")
proc dtype*(T: typedesc[float64]): PyObject = nptypes("double")
proc dtype*(T: typedesc[Complex32]): PyObject = nptypes("csingle")
proc dtype*(T: typedesc[Complex64]): PyObject = nptypes("cdouble")

type
  NumpyArray*[T] = PyObject

proc assertNumpyType[T](ndArray: PyObject) =
  let
    dtype_sizeof = dtype(ndArray).itemsize.to(int)*sizeof(byte)
    dtype_kind = dtype(ndArray).kind.to(string)[0]

  if sizeof(T) != dtype_sizeof:
    raiseAssert(&"Error converting PyObject NDArray to Arraymancer Tensor. Type sizeof({$T})={sizeof(T)} not equal to numpy.dtype.itemsize ({dtype_sizeof}).")

  let msg = &"Error converting PyObject NDArray to Arraymancer Tensor. Type {$T} not compatible with numpy.dtype.kind {dtype_kind}."
  when T is SomeFloat:
    if dtype_kind != 'f':
      raiseAssert(msg)

  elif T is SomeSignedInt:
    if dtype_kind != 'i':
      raiseAssert(msg)

  elif T is SomeUnsignedInt:
    if dtype_kind != 'u':
      raiseAssert(msg)

  elif T is bool:
    if dtype_kind != 'b':
      raiseAssert(msg)

  else:
    raiseAssert(msg)

proc asNumpyArray*[T](o: PyObject): NumpyArray[T] {.inline.} =
  assertNumpyType[T](o)
  return NumpyArray[T](o)

proc numpyArrayToTensor[T](ndArray: NumpyArray[T]): Tensor[T] =
  # Get buffer of PyObject
  var aBuf: RawPyBuffer
  getBuffer(ndArray.PyObject, aBuf, PyBUF_WRITABLE or PyBUF_ND)
  # Get shape and size in bytes of numpy array
  let
    nbytes = ndArray.nbytes.to(int)
    shape = (ndArray.shape).to(seq[int])
  # Alloc tensor
  # No need for sanity check here
  result = newTensor[T](shape)
  copyMem(toUnsafeView(result), aBuf.buf, nbytes)
  aBuf.release()

proc toTensor*[T](ndArray: NumpyArray[T]): Tensor[T] =
  result = numpyArrayToTensor[T](ndArray)

proc ndArrayFromPtr*[T](t: ptr T, nelem: int): NumpyArray[T] =
  let np = pyImport("numpy")
  let py_array_type = dtype(T)

  result = NumpyArray[T](nimpy.callMethod(np, "zeros", nelem, py_array_type))
  var aBuf: RawPyBuffer
  getBuffer(result.PyObject, aBuf, PyBUF_WRITABLE or PyBUF_ND)
  var bsizes = nelem*(sizeof(T) div sizeof(uint8))
  copyMem(aBuf.buf, t, bsizes)
  aBuf.release()

# Convert Tensor to RawPyBuffer
proc ndArrayFromTensor[T](t: Tensor[T]): NumpyArray[T] =
  # Reshape PyObject to Arraymancer Tensor
  let np = pyImport("numpy")
  var s = t.shape.toSeq()
  var shape = ndArrayFromPtr(addr(s[0]), s.len())
  var ndArray = ndArrayFromPtr[T](cast[ptr T](toUnsafeView(t)), t.size)

  # Reshape numpy array
  result = NumpyArray[T](nimpy.callMethod(np, "reshape", ndArray, shape))

proc toNdArray*[T](t: Tensor[T]): NumpyArray[T] =
  ndArrayFromTensor[T](t)
