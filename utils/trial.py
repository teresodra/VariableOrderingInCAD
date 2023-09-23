import sympy as sp


def create_polynomial(monomials):
    """
    Create a polynomial from a list of monomials, where each monomial is represented as [degrees, coefficient].

    Args:
    - monomials (list of lists): List of monomials, where each monomial is represented as [degrees, coefficient].

    Returns:
    - sympy expression: The polynomial expression.
    """

    # Make sure the list of monomials is not empty
    if monomials == []:
        return 0

    # Compute the number of variables
    num_variables = len(monomials[0]) - 1

    # Create symbolic variables in a loop and store them in a list
    variables = [sp.symbols(f'x{i}') for i in range(num_variables)]

    polynomial = sum(coefficient * sp.prod(var**degree
                     for var, degree in zip(variables, degrees))
                     for *degrees, coefficient in monomials)

    return polynomial

# Example usage:
# if __name__ == "__main__":

#     # List of monomials, each represented as [degrees, coefficient]
#     monomials = [[1, 2, 0, 1], [1, 2, 3, 4]]

#     # Create the polynomial
#     polynomial = create_polynomial(monomials)

#     # Print the polynomial
#     print("Polynomial:", polynomial)

dict1 = {'a': 2, 'b': 1, 'c': 3}
dict2 = {'b': 1, 'a': 2, 'c': 3}
assertDictEqual(dict1, dict2)