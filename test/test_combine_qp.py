#!/usr/bin/env python3

import torch
import sys
sys.path.append('/home/ReLUQP-py/wbc_ik_qp')
from ho_qp import Task, HoQp

def test_orthogonal():
    print("Testing HoQp with Orthogonal Problems")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    # Problem 1: x[0:3] = [1, 2, 3]
    A1 = torch.zeros((3, 6), device=device, dtype=dtype)
    A1[0, 0] = 1.0
    A1[1, 1] = 1.0  
    A1[2, 2] = 1.0
    b1 = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    task1 = Task(a=A1, b=b1, device=device, dtype=dtype, weight=1.0)
    
    # Problem 2: x[3:6] = [4, 5, 6]
    A2 = torch.zeros((3, 6), device=device, dtype=dtype)
    A2[0, 3] = 1.0
    A2[1, 4] = 1.0
    A2[2, 5] = 1.0
    b2 = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)
    task2 = Task(a=A2, b=b2, device=device, dtype=dtype, weight=1.0)
    
    print("Problem 1 (High): x[0:3] = [1, 2, 3]")
    print("Problem 2 (Low):  x[3:6] = [4, 5, 6]")
    
    # Individual solutions
    ho1 = HoQp(task1, higher_problem=None, device=device, dtype=dtype)
    sol1 = ho1.getSolutions()
    print(f"Problem 1 alone: {sol1}")
    
    ho2 = HoQp(task2, higher_problem=None, device=device, dtype=dtype)
    sol2 = ho2.getSolutions()
    print(f"Problem 2 alone: {sol2}")
    
    # Combined solution
    combined = HoQp(task2, higher_problem=ho1, device=device, dtype=dtype)
    result = combined.getSolutions()
    print(f"Combined result: {result}")
    
    # Check if result matches expected [1, 2, 3, 4, 5, 6]
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=device, dtype=dtype)
    error = torch.norm(result - expected).item()
    print(f"Expected: {expected}")
    print(f"Error: {error:.8f}")
    
    success = error < 1e-6
    print(f"Test: {'PASS' if success else 'FAIL'}")
    return success

def test_competing():
    print("\nTesting HoQp with Competing Problems")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    # Both problems want different values for the SAME variables x[0:2]
    # Problem 1 (High Priority): x[0] = 10, x[1] = 20
    A1 = torch.zeros((2, 4), device=device, dtype=dtype)
    A1[0, 0] = 1.0  # x[0] = 10
    A1[1, 1] = 1.0  # x[1] = 20
    b1 = torch.tensor([10.0, 20.0], device=device, dtype=dtype)
    task1 = Task(a=A1, b=b1, device=device, dtype=dtype, weight=1.0)
    
    # Problem 2 (Low Priority): x[0] = 5, x[1] = 15 (CONFLICTS with Problem 1!)
    A2 = torch.zeros((2, 4), device=device, dtype=dtype)
    A2[0, 0] = 1.0  # x[0] = 5  (conflicts with Problem 1)
    A2[1, 1] = 1.0  # x[1] = 15 (conflicts with Problem 1)
    b2 = torch.tensor([5.0, 15.0], device=device, dtype=dtype)
    task2 = Task(a=A2, b=b2, device=device, dtype=dtype, weight=1.0)
    
    print("Problem 1 (High): x[0] = 10, x[1] = 20")
    print("Problem 2 (Low):  x[0] = 5,  x[1] = 15  (COMPETING!)")
    print("Expected: High priority should win")
    
    # Individual solutions
    ho1 = HoQp(task1, higher_problem=None, device=device, dtype=dtype)
    sol1 = ho1.getSolutions()
    print(f"Problem 1 alone: {sol1}")
    
    ho2 = HoQp(task2, higher_problem=None, device=device, dtype=dtype)
    sol2 = ho2.getSolutions()
    print(f"Problem 2 alone: {sol2}")
    
    # Combined solution - high priority should dominate
    combined = HoQp(task2, higher_problem=ho1, device=device, dtype=dtype)
    result = combined.getSolutions()
    print(f"Combined result: {result}")
    
    # Check if high priority wins: x[0] should be 10, x[1] should be 20
    expected = torch.tensor([10.0, 20.0, 0.0, 0.0], device=device, dtype=dtype)
    error = torch.norm(result - expected).item()
    print(f"Expected: {expected}")
    print(f"Error: {error:.8f}")
    
    # Test that high priority constraints are satisfied
    high_priority_error = torch.norm(result[:2] - torch.tensor([10.0, 20.0], device=device, dtype=dtype)).item()
    success = high_priority_error < 1e-6
    print(f"High priority satisfied: {'PASS' if success else 'FAIL'}")
    print(f"Test: {'PASS' if success else 'FAIL'}")
    return success

def test_45_degree():
    print("\nTesting HoQp with 45-Degree Angled Problems")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    # Create two tasks that operate in 45-degree rotated spaces
    # Task 1 (High Priority): constraint in direction [1, 1] 
    # This means x[0] + x[1] = 10 (a line at 45 degrees)
    A1 = torch.zeros((1, 3), device=device, dtype=dtype)
    A1[0, 0] = 1.0  # x[0] + x[1] = 10
    A1[0, 1] = 1.0
    b1 = torch.tensor([10.0], device=device, dtype=dtype)
    task1 = Task(a=A1, b=b1, device=device, dtype=dtype, weight=1.0)
    
    # Task 2 (Low Priority): constraint in direction [1, -1]
    # This means x[0] - x[1] = 2 (a line at -45 degrees)  
    # These two constraints would normally intersect at x[0]=6, x[1]=4
    A2 = torch.zeros((1, 3), device=device, dtype=dtype)
    A2[0, 0] = 1.0   # x[0] - x[1] = 2  
    A2[0, 1] = -1.0
    b2 = torch.tensor([2.0], device=device, dtype=dtype)
    task2 = Task(a=A2, b=b2, device=device, dtype=dtype, weight=1.0)
    
    print("Problem 1 (High): x[0] + x[1] = 10  (45° constraint)")
    print("Problem 2 (Low):  x[0] - x[1] = 2   (-45° constraint)")
    print("If both satisfied: x[0]=6, x[1]=4")
    print("Expected: High priority constraint satisfied, low priority projected to nullspace")
    
    # Individual solutions
    ho1 = HoQp(task1, higher_problem=None, device=device, dtype=dtype)
    sol1 = ho1.getSolutions()
    print(f"Problem 1 alone: {sol1}")
    print(f"  Check: x[0]+x[1] = {sol1[0] + sol1[1]:.6f} (should be 10)")
    
    ho2 = HoQp(task2, higher_problem=None, device=device, dtype=dtype)
    sol2 = ho2.getSolutions()
    print(f"Problem 2 alone: {sol2}")
    print(f"  Check: x[0]-x[1] = {sol2[0] - sol2[1]:.6f} (should be 2)")
    
    # Combined solution
    combined = HoQp(task2, higher_problem=ho1, device=device, dtype=dtype)
    result = combined.getSolutions()
    print(f"Combined result: {result}")
    
    # Verify constraints
    high_constraint = result[0] + result[1]  # Should be 10
    low_constraint = result[0] - result[1]   # Will be projected
    
    print(f"  High priority: x[0]+x[1] = {high_constraint:.6f} (should be 10)")
    print(f"  Low priority:  x[0]-x[1] = {low_constraint:.6f} (projected)")
    
    # Test that high priority constraint is satisfied exactly
    high_error = abs(high_constraint - 10.0)
    success = high_error < 1e-6
    
    print(f"High priority error: {high_error:.8f}")
    print(f"High priority satisfied: {'PASS' if success else 'FAIL'}")
    
    # Additional analysis: show the nullspace projection effect
    # The nullspace of [1, 1, 0] is the plane perpendicular to [1, 1, 0]
    # The low priority task [1, -1, 0] projected onto this nullspace should show
    # how much of the low priority constraint can be satisfied
    print(f"\nAnalysis:")
    print(f"  - High priority defines: x[0] + x[1] = 10")
    print(f"  - Nullspace allows variation in direction [1, -1, 0] and [0, 0, 1]")
    print(f"  - Low priority tries to enforce x[0] - x[1] = 2")
    print(f"  - Result shows compromise in the nullspace")
    
    print(f"Test: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    try:
        test1 = test_orthogonal()
        test2 = test_competing()
        test3 = test_45_degree()
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print(f"Orthogonal Test:  {'PASS' if test1 else 'FAIL'}")
        print(f"Competing Test:   {'PASS' if test2 else 'FAIL'}")
        print(f"45-Degree Test:   {'PASS' if test3 else 'FAIL'}")
        
        if test1 and test2 and test3:
            print("🎉 HoQp module is working correctly!")
        else:
            print("⚠️  HoQp module has issues!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
