"""
Test if the generated input file can be imported
"""
import sys
import os

# Change to the directory where the generated file is
test_dir = 'test_input_gen'
os.chdir(test_dir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..', 'simulator'))

print(f"Current directory: {os.getcwd()}")
print(f"Python path includes:")
for p in sys.path[:3]:
    print(f"  - {p}")

print("\nAttempting to import the generated input file...")

try:
    import input_sample_test
    print("\n✓ Import successful!")
    print(f"Numerical.tfinal = {input_sample_test.numerical.tfinal}")
    print(f"Number of wells = {len(input_sample_test.well.x)}")
except Exception as e:
    print(f"\n✗ Import failed!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
