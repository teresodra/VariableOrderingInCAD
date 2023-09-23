import sympy as sp
import numpy as np
from typing import List, Dict


def newton_raphson(equations: List[sp.Expr],
                   variables: List[sp.Symbol],
                   initial_guess: List[float],
                   tol: float = 1e-6,
                   max_iter: int = 100,
                   perturb_factor: float = 1e-5) -> List[float]:
    """
    Approximate the root of a set of multivariate polynomials
    using the Newton-Raphson method.

    Args:
    - equations (list of sympy expressions): The system of equations to solve.
    - variables (list of sympy symbols): The variables in the equations.
    - initial_guess (list of float): Initial guess for the root.
    - tol (float): Tolerance for convergence.
    - max_iter (int): Maximum number of iterations.
    - perturb_factor (float): Factor to perturb the root when the
    Jacobian determinant is close to zero.

    Returns:
    - list of float: Approximated root.
    """

    # print(equations, variables, initial_guess)
    m_vars = sp.Matrix(variables)
    f = sp.Matrix(equations)
    J = f.jacobian(m_vars)
    root = sp.Matrix(initial_guess)

    for i in range(max_iter):
        f_eval = [float(expr.subs(list(zip(m_vars, root)))) for expr in f]

        if all(abs(val) < tol for val in f_eval):
            # print('max_iter', i)
            return [float(val) for val in root]

        J_eval = J.subs(list(zip(m_vars, root)))
        determinant = J_eval.det()

        # Set a counter
        tries = 0
        while abs(determinant) < tol and tries < max_iter:
            tries += 1
            # print(tries, max_iter)
            # Perturb the root slightly
            perturbation = np.random.rand(len(variables)) * perturb_factor
            root += sp.Matrix(perturbation)
            J_eval = J.subs(list(zip(m_vars, root)))
            determinant = J_eval.det()

        # Exit if the determinant is still to low
        if tries >= max_iter:
            break

        delta_m_vars = -J_eval.inv() * sp.Matrix(f_eval)
        root += delta_m_vars

    raise ValueError("Newton-Raphson method did not converge within "
                     "the specified maximum number of iterations.")


def find_roots(equations: List[sp.Expr],
               variables: List[sp.Symbol],
               tol: float = 1e-6,
               trials_factor: int = 5,
               min_trials: int = 10) -> List[List[float]]:
    """
    Find roots of a set of multivariate polynomials
    using the Newton-Raphson method
    with automatic initial guess generation and
    stopping criteria based on equation characteristics.

    Args:
    - equations (list of sympy expressions): The system of equations to solve.
    - variables (list of sympy symbols): The variables in the equations.
    - tol (float): Tolerance for convergence.
    - trials_factor (int): The higher, the more times we look for new roots
    - min_trials (int): Minimum number of times we look for new roots

    Returns:
    - list of lists of float: Approximated roots.
    """

    if len(equations) != len(variables):
        raise ValueError("The number of independent equations should be equal"
                         "to the number of variables to have finite roots.")

    roots = []
    attempts = 0

    # Choose the range for the initial guesses
    guess_range = 0
    # Choose maximum number of attempts to find new roots
    max_attempts = min_trials
    for polynomial in equations:
        # Compute the total degree of the polynomial
        total_degree = polynomial.as_poly().total_degree()
        if total_degree != 0:
            abs_indep_coeff = abs(polynomial.as_coefficients_dict()[1])
            guess_range = max(guess_range,
                              2 * abs_indep_coeff ** (1 / total_degree))
            max_attempts = total_degree * trials_factor

    while attempts < max_attempts:
        # Generate a random initial guess
        initial_guess = [np.random.uniform(-guess_range, guess_range)
                         for _ in variables]
        # print('initial_guess', initial_guess)

        try:
            root = newton_raphson(equations, variables, initial_guess, tol)
            # print('root', root)

            # Check if the root is new (within tolerance)
            is_new = all(
                np.linalg.norm(np.array(root) - np.array(existing_root)) > tol
                for existing_root in roots
            )

            if is_new:
                roots.append(root)
                attempts = 0  # Reset attempts if a new root is found
            else:
                # print('not new')
                attempts += 1

        except ValueError:
            # Count the attempt if convergence error for this initial guess
            attempts += 1

    return roots


def critical_points(polynomial: sp.Expr,
                    trials_factor: int = 5) -> Dict[sp.Symbol, int]:
    '''
    Compute the critical points of the given polynomial with respect
    to each of its variables.

    Args:
    - polynomial (sympy expression): The polynomial expression.
    - trials_factor (int): The higher, the more times we look for new roots

    Returns:
    - dict: A dictionary containing the number of critical points
            for each variable in the polynomial.
    '''

    # Find its variables
    variables = list(polynomial.free_symbols)

    # Store derivatives
    derivatives = {}
    for var in variables:
        derivatives[var] = sp.diff(polynomial, var)

    # Dictionary to store number of critical points
    critical_points = dict()
    for var in variables:
        # print('var', var)
        # List of equations containing:
        # derivatives with respect to other variables
        # and the original polynomial
        equations = [derivatives[aux_var] for aux_var in variables
                     if aux_var != var] + [polynomial]

        # Use a root-finding function to find the critical points,
        # that are the common roots of the equations
        roots = find_roots(equations, variables, trials_factor=trials_factor)
        # print('roots', roots)
        critical_points[var] = len(roots)

    return critical_points


def count_critical_points(polynomials: List[sp.Expr]) -> Dict[sp.Symbol, int]:
    """
    Compute the total count of critical points
    for each variable in a set of polynomials.

    Args:
    - polynomials (list of sympy expressions):
    The set of multivariate polynomials to analyze.

    Returns:
    - dict: A dictionary where keys are variables
    and values are the count of critical points.
    """

    # Find all unique variables present in the set of polynomials
    unique_variables = set()
    for polynomial in polynomials:
        unique_variables.update(polynomial.free_symbols)

    # Initialize a dictionary to store the total count of critical points
    # for each variable
    total_critical_points = {variable: 0 for variable in unique_variables}

    # Iterate through each polynomial to find critical points
    # and update the counts
    for polynomial in polynomials:
        for variable, count in critical_points(polynomial).items():
            total_critical_points[variable] += count

    return total_critical_points


# Example usage:
if __name__ == "__main__":
    # Define a set of multivariate polynomials
    x, y = sp.symbols('x y')
    polynomials = [x**2 + y**2 - 4, x**3 * y - 4 * y**3]
    polynomials = []

    # Compute the total count of critical values for each variable
    critical_values = count_critical_points(polynomials)
    print(critical_values)  # Example output: {y: 3, x: 3}
