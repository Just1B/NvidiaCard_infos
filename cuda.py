#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ctypes
import os

# Constantes récupérées du fichier info.cs
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37


def ConvertSMVer2Cores(major, minor):
    return {(1, 0): 8,
            (1, 1): 8,
            (1, 2): 8,
            (1, 3): 8,
            (2, 0): 32,
            (2, 1): 48,
            }.get((major, minor), 192)


def main():
    os.system('nvidia-smi')

    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    bandwidth = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString( result, ctypes.byref(error_str) )
        print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    result = cuda.cuDeviceGetCount( ctypes.byref(nGpus) )
    
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString( result, ctypes.byref(error_str) )
        print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    print(" %d carte(s) détectée(s)." % nGpus.value)
    
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet( ctypes.byref(device), i )
       
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString( result, ctypes.byref(error_str) )
            print("cuDeviceGet failed with error code %d: %s" % (result, error_str.value.decode()))
            return 1
        print("\n Carte n° : %d" % i)
       
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            print("  Nom : %s" % (name.split(b'\0', 1)[0].decode()))
        
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            print("  Capacité de calcul : %d.%d" % (cc_major.value, cc_minor.value))
       
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            print("  Multiprocesseurs : %d" % cores.value)
            print("  Cores CUDA : %d" % (cores.value * ConvertSMVer2Cores(cc_major.value, cc_minor.value)))
           
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                print("  Unités de calcul : %d" % ( threads_per_core.value))
       
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
            print("  Fréquence du GPU : %g MHz" % ( clockrate.value / 1000.) )
        
        if cuda.cuDeviceGetAttribute( ctypes.byref(bandwidth), CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device) == CUDA_SUCCESS:
            print("  Bande passante : %g GB/s " % ( ( bandwidth.value ) ) )
        result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
       
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
        else:
            result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                print("  Mémoire Totale : %ld MiB" % (totalMem.value / 1024**2))
                print("  Mémoire libre : %ld MiB" % (freeMem.value / 1024**2))
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                print("cuMemGetInfo failed with error code %d: %s" % (result, error_str.value.decode()))
            cuda.cuCtxDetach(context)
    return 0


if __name__=="__main__":
    sys.exit(main())