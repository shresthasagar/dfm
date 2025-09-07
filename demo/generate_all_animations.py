#!/usr/bin/env python3
"""
Script to generate animations for all flow matching methods in both 2D and 3D.
This creates a comprehensive comparison gallery for the README.
"""

import subprocess
import os
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ Completed successfully in {elapsed:.1f}s")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return False

def main():
    """Generate all animations for methods comparison."""
    
    # Configuration
    methods = ['fm', 'fm_ot', 'dfm']
    configs = ['2d', '3d']
    seed = 42
    
    method_names = {
        'fm': 'Standard Flow Matching',
        'fm_ot': 'Flow Matching + Optimal Transport', 
        'dfm': 'Diversified Flow Matching (Ours)'
    }
    
    # Ensure we're in the synthetic directory
    if not os.path.exists('synthetic_demo.py'):
        print("❌ Error: synthetic_demo.py not found. Please run this script from the synthetic/ directory.")
        sys.exit(1)
    
    print("🎬 Flow Matching Animation Gallery Generator")
    print("="*60)
    print("This script will generate animations for:")
    for method in methods:
        print(f"  • {method_names[method]} ({method})")
    print(f"Configurations: {', '.join(configs)}")
    print(f"Seed: {seed}")
    
    # Ask for confirmation
    response = input("\nProceed with animation generation? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)
    
    total_runs = len(methods) * len(configs)
    current_run = 0
    successful_runs = 0
    failed_runs = []
    
    start_total = time.time()
    
    print(f"\n🎯 Starting {total_runs} animation generations...")
    
    # Generate animations for each method and configuration
    for config in configs:
        print(f"\n🔄 Processing {config.upper()} configurations...")
        
        for method in methods:
            current_run += 1
            
            # Build command
            cmd = [
                sys.executable, 'synthetic_demo.py',
                '--method', method,
                '--config', config,
                '--seed', str(seed),
                '--create_animations'
            ]
            
            description = f"[{current_run}/{total_runs}] {method_names[method]} - {config.upper()}"
            
            # Run the command
            success = run_command(cmd, description)
            
            if success:
                successful_runs += 1
                
                # Check if expected files were created
                result_dir = f"results/{method}_dim{config[-1]}_seed{seed}"
                animation_file = f"{result_dir}/final_test_{config}_animation.gif"
                
                if os.path.exists(animation_file):
                    file_size = os.path.getsize(animation_file) / (1024 * 1024)  # MB
                    print(f"📁 Animation saved: {animation_file} ({file_size:.1f} MB)")
                else:
                    print(f"⚠️  Warning: Expected animation file not found: {animation_file}")
            else:
                failed_runs.append(f"{method} {config}")
    
    # Summary
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print("🎉 ANIMATION GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful_runs}/{total_runs}")
    print(f"❌ Failed: {len(failed_runs)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    if failed_runs:
        print(f"\nFailed runs:")
        for run in failed_runs:
            print(f"  • {run}")
    
    # List generated files
    print(f"\n📂 Generated files:")
    results_dir = Path("results")
    if results_dir.exists():
        for method in methods:
            for config in configs:
                pattern = f"{method}_dim{config[-1]}_seed{seed}"
                method_dir = results_dir / pattern
                if method_dir.exists():
                    animation_file = method_dir / f"final_test_{config}_animation.gif"
                    if animation_file.exists():
                        size_mb = animation_file.stat().st_size / (1024 * 1024)
                        print(f"  ✅ {animation_file} ({size_mb:.1f} MB)")
                    else:
                        print(f"  ❌ {animation_file} (missing)")
    
    # Next steps
    print(f"\n🔗 Next steps:")
    print(f"1. Check the generated animations in the results/ directory")
    print(f"2. Update the README.md to include the new animation paths")
    print(f"3. Commit the new animations to your repository")
    
    if successful_runs == total_runs:
        print(f"\n🎊 All animations generated successfully!")
        return True
    else:
        print(f"\n⚠️  Some animations failed to generate. Check the errors above.")
        return False

if __name__ == "__main__":
    main()
