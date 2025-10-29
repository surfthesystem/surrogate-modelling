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

# Get absolute paths
run_dir = os.path.abspath(os.path.dirname(test_file))
simulator_dir = os.path.abspath('../simulator')

print(f"Run directory: {run_dir}")
print(f"Simulator directory: {simulator_dir}")
print(f"Input file exists: {os.path.exists(os.path.join(run_dir, 'input_file_phase1.py'))}")

# Change to run directory so relative paths work
os.chdir(run_dir)
print(f"Changed working directory to: {os.getcwd()}")

# Add to path - run_dir MUST come before simulator_dir!
# Otherwise Python will import the template instead of the generated file
sys.path.insert(0, simulator_dir)  # Insert simulator first
sys.path.insert(0, run_dir)  # Then insert run_dir at position 0 (so it's checked first)

print(f"sys.path[0]: {sys.path[0]}")
print(f"sys.path[1]: {sys.path[1]}")

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
