def expected_degree(n: int, p: float) -> float:
    """
    Calculate the expected degree of a node in an Erdős-Rényi random graph.

    Args:
        n (int): The number of nodes in the graph.
        p (float): The probability of an edge existing between any two nodes.

    Returns:
        float: The expected degree of a node.
    """
    return (n - 1) * p  # Subtract 1 since there are no self-loops

def test_expected_degree():
    """
    Test the expected degree calculation for specific parameters.
    """
    n = 15
    p1 = 0.1
    p2 = 0.4

    # Expected degrees
    expected_degree_p1 = expected_degree(n, p1)
    expected_degree_p2 = expected_degree(n, p2)

    print(f"Expected degree for G(15, 0.1): {expected_degree_p1}")
    print(f"Expected degree for G(15, 0.4): {expected_degree_p2}")

if __name__ == "__main__":
    print("Testing expected degree calculation...")
    test_expected_degree()
