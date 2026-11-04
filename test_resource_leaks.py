#!/usr/bin/env python3
"""
Test script to demonstrate potential resource leaks in PyEBSDIndex.
This script tests file descriptor and memory usage during repeated operations.
"""

import os
import gc
import psutil
from pathlib import Path
import numpy as np

# Import the problematic modules
try:
    from pyebsdindex.ebsd_pattern import EBSDPatterns
except ImportError as e:
    print(f"Failed to import EBSDPatterns: {e}")
    exit(1)


def get_open_files():
    """Get count of open file descriptors for current process."""
    proc = psutil.Process(os.getpid())
    try:
        return len(proc.open_files())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return -1


def get_memory_usage():
    """Get memory usage in MB."""
    proc = psutil.Process(os.getpid())
    try:
        return proc.memory_info().rss / 1024 / 1024
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return -1


def test_file_descriptor_leak():
    """Test for file descriptor leaks in EBSDPatterns."""
    print("Testing for file descriptor leaks...")
    
    initial_files = get_open_files()
    print(f"Initial open files: {initial_files}")
    
    # Test multiple operations that should close files
    for i in range(10):
        try:
            # Create a temporary test file
            test_file = Path("/tmp/test_pattern.up1")
            
            # Create test pattern data
            if not test_file.exists():
                pattern_data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
                pattern_data.tofile(test_file)
            
            # Test opening and closing
            patterns = EBSDPatterns(str(test_file))
            
            # Try to read some data
            patterns.nPatterns = 10
            patterns.patternW = 10
            patterns.patternH = 10
            
            # This should trigger file operations that might leak descriptors
            try:
                patterns.readpatterns(patstart=0, npats=1)
            except Exception as e:
                print(f"Error reading patterns: {e}")
            
            # Explicitly close if possible
            if hasattr(patterns, 'fID') and patterns.fID:
                try:
                    patterns.fID.close()
                except:
                    pass
            
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
                
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
        
        # Force garbage collection
        gc.collect()
    
    final_files = get_open_files()
    print(f"Final open files: {final_files}")
    
    leak_detected = final_files > initial_files + 5
    print(f"File descriptor leak detected: {leak_detected}")
    
    return leak_detected


def test_memory_usage():
    """Test memory usage during repeated operations."""
    print("\nTesting memory usage...")
    
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    for i in range(5):
        try:
            # Create large temporary arrays
            large_array = np.random.rand(1000, 1000)
            
            # Force cleanup
            del large_array
            gc.collect()
            
        except Exception as e:
            print(f"Error in memory test iteration {i}: {e}")
    
    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.1f} MB")
    
    increase_detected = final_memory > initial_memory + 50
    print(f"Memory usage increase detected: {increase_detected}")
    
    return increase_detected


if __name__ == "__main__":
    print("PyEBSDIndex Resource Leak Test")
    print("=" * 50)
    
    # Check if psutil is available
    try:
        import psutil
        print("psutil available - can monitor resources")
    except ImportError:
        print("psutil not available - cannot monitor resources properly")
        exit(1)
    
    # Run tests
    fd_leak = test_file_descriptor_leak()
    mem_increase = test_memory_usage()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"File descriptor leak: {'YES - CRITICAL' if fd_leak else 'No issues detected'}")
    print(f"Memory usage increase: {'YES - CONCERNING' if mem_increase else 'No issues detected'}")
    
    if fd_leak or mem_increase:
        print("\n⚠️  Potential resource leaks detected!")
        print("Recommend running with increased file descriptor limits and monitoring memory usage.")
    else:
        print("\n✅ No obvious resource leaks detected in basic tests.")