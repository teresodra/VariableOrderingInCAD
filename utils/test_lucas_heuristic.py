import sympy as sp
import unittest
import time

# Import the critical_points function from your module
from lucas_heuristic import critical_points
from lucas_heuristic import find_roots
from lucas_heuristic import newton_raphson
from lucas_heuristic import count_critical_points


class TestCountCriticalPoints(unittest.TestCase):

    def test_empty_input(self):
        # Test with an empty list of polynomials
        result = count_critical_points([])
        expected = dict()
        self.assertEqual(result, expected)

    def test_single_polynomial(self):
        # Test with a single polynomial with known critical points
        x, y = sp.symbols('x y')
        polynomial = x**2 + y**2 - 4
        result = count_critical_points([polynomial])
        expected = {x: 2, y: 2}  # Expected number of critical points
        self.assertEqual(result, expected)

    def test_multiple_polynomials(self):
        # Test with multiple polynomials with known critical points
        x, y = sp.symbols('x y')
        polynomials = [x**2 + y**2 - 4,
                       x**3 * y - 4 * y**3]
        result = count_critical_points(polynomials)
        expected = {x: 3, y: 3}  # Expected number of critical points
        self.assertEqual(result, expected)


class TestNewtonRaphson(unittest.TestCase):

    def test_valid_input(self):
        # Test with a valid set of equations, variables, and initial guess
        x, y = sp.symbols('x y')
        equations = [x**2 - 4,
                     y**2 - 9]  # Two simple equations with roots (+-2, +-3)
        variables = [x, y]
        initial_guess = [1, 1]
        root = newton_raphson(equations, variables, initial_guess)

        # Ensure the root is one of the possibilities
        self.assertAlmostEqual(root[0], 2, msg="The root given is not a root")
        self.assertAlmostEqual(root[1], 3, msg="The root given is not a root")

    def test_invalid_input(self):
        # Test with invalid input where the number of equations
        # and variables don't match
        x, y, z = sp.symbols('x y z')
        equations = [x**2 - 4, y**2 - 9]  # Two equations, but three variables
        variables = [x, y, z]
        initial_guess = [1, 1]

        with self.assertRaises(ValueError):
            newton_raphson(equations, variables, initial_guess)

    def test_non_converging_case(self):
        # Test a case where Newton-Raphson method
        # doesn't converge within max_iter
        x = sp.symbols('x')
        equations = [x**2 + 1]  # No real roots
        variables = [x]
        initial_guess = [1]

        with self.assertRaises(ValueError):
            newton_raphson(equations, variables, initial_guess, max_iter=10)

    def test_not_infinite_loop(self):
        # It was observed that there can be an infinite loop
        # this was fixed but example added here
        # this test takes more than 10s
        # if it enters the almost infinite loop

        # Start measuring time
        start_time = time.time()

        x, y = sp.symbols('x y')
        equations = [x**3, x**3*y - 4]
        variables = [x, y]
        initial_guess = [-1, 0]

        with self.assertRaises(ValueError):
            newton_raphson(equations, variables, initial_guess)

        # Total time taken
        execution_time = time.time() - start_time
        # Assert that the execution time is less than 1 second
        self.assertLess(execution_time, 1.0, "Test took longer than 1 second")


class TestFindRoots(unittest.TestCase):

    def test_valid_equations(self):
        # Test with valid equations and variables
        x, y = sp.symbols('x y')
        equations = [x**2 + y**2 - 1,  # Circle equation
                     x - y]            # diagonal line
        variables = [x, y]
        tol = 1e-6
        roots = find_roots(equations, variables, tol=tol)

        # Ensure the number of roots is correct (should be 2)
        self.assertEqual(len(roots), 2)

        # Ensure that they are the correct roots
        sqrt2div2 = 0.7071068
        for root in roots:
            for coordinate in root:
                self.assertAlmostEqual(abs(coordinate), sqrt2div2)

    def test_invalid_input(self):
        # Test with invalid input; number of equations
        # and variables dont match
        x, y, z = sp.symbols('x y z')
        equations = [x**2 - 4, y**2 - 9]  # Two equations
        variables = [x, y, z]             # but three variables

        with self.assertRaises(ValueError):
            find_roots(equations, variables)

    def test_no_roots(self):
        # Test with an invalid equation that cannot be solved
        x, y = sp.symbols('x y')
        equations = [x**2 + y**2 + 1,  # No real roots
                     x - y]
        variables = [x, y]
        roots = find_roots(equations, variables)

        # Ensure no roots are found (empty list)
        self.assertEqual(len(roots), 0)


class TestCriticalPoints(unittest.TestCase):

    def test_single_variable(self):
        # Test a polynomial with a single variable (e.g., x)
        x = sp.symbols('x')
        poly = x**2 + 3*x + 2
        result = critical_points(poly)
        expected = {x: 2}
        self.assertDictEqual(result, expected)

    def test_two_variables(self):
        # Test a polynomial with two variables
        x, y = sp.symbols('x y')
        poly = x**2 + y**2 - 4
        result = critical_points(poly)
        expected = {x: 2, y: 2}
        self.assertDictEqual(result, expected)

    def test_no_independent_coefficient(self):
        # Test a polynomial with no critical points
        x, y = sp.symbols('x y')
        poly = x**2 + y**2 * x
        result = critical_points(poly)
        expected = {x: 1, y: 1}
        self.assertDictEqual(result, expected)

    def test_no_critical_points(self):
        # Test a polynomial with no critical points
        x, y = sp.symbols('x y')
        poly = x**2 + y**2 + 1
        result = critical_points(poly)
        expected = {x: 0, y: 0}
        self.assertDictEqual(result, expected)

    def test_complex_critical_points(self):
        # Test a polynomial with complex critical points
        x, y, z, w = sp.symbols('x y z w')
        poly = x**4 + y**4 + z**4 + w**4 - 6*x**2 - 6*y**2 - 6*z**2 - 6*w**2
        result = critical_points(poly, 10)
        expected = {x: 1, y: 1, z: 1, w: 1}
        self.assertDictEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
