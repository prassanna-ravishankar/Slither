#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'slither_build')

try:
    from slither_py import slither
    print("✅ Python module imports successfully")
    
    # Create a simple instance
    wrapper = slither()
    print("✅ SlitherWrapper instance created")
    
    # Test a simple method
    result = wrapper.add(2, 3)
    print(f"✅ Simple method call works: 2 + 3 = {result}")
    
    # Check if we can set default parameters
    wrapper.setDefaultParams()
    print("✅ Default parameters set")
    
    print("\n🎉 All basic Python binding tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)