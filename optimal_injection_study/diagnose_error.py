"""
Diagnostic script to see the exact error when importing generated input files
"""
import sys
import os
import traceback

# Try to import a generated input file and show the full error
test_file = 'lhs_results/run_000/input_file_phase1.py'

if not os.path.exists(test_file):
    print(f"Error: {test_file} does not exist")
    print("Please run: python run_lhs_optimization.py")
    print("to generate at least one input file first")
    sys.exit(1)

print(f"Attempting to import: {test_file}")
print("="*70)

# Change to the directory
run_dir = os.path.dirname(test_file)
sys.path.insert(0, run_dir)
sys.path.insert(0, '../../simulator')

os.chdir(run_dir)

try:
    import input_file_phase1
    print("SUCCESS! File imported without errors")
    print(f"Number of wells: {len(input_file_phase1.well.x)}")
    print(f"Simulation time: {input_file_phase1.numerical.tfinal} days")
except Exception as e:
    print("FAILED! Here's the full error:")
    print("="*70)
    traceback.print_exc()
    print("="*70)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {e}")

    # Try to show the problematic line
    if hasattr(e, 'lineno'):
        print(f"\nError at line: {e.lineno}")
        with open('input_file_phase1.py', 'r') as f:
            lines = f.readlines()
            if e.lineno <= len(lines):
                print(f"Line {e.lineno}: {lines[e.lineno-1]}")
