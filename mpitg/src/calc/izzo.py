import numpy as np

from mpitg.src.almanac.constants import AU, MU_SUN


# Function to compute the chord distance between two vectors
def chord_distance(r1, r2):
    return np.linalg.norm(r2 - r1)

# Function to compute the semi-perimeter of the triangle
def semi_perimeter(r1, r2, c):
    return (np.linalg.norm(r1) + np.linalg.norm(r2) + c) / 2

# Function to compute the initial guess for Lambert's problem using Izzo's method
def izzo_initial_guess(r1, r2, delta_t):
    # Convert distances from AU to meters (if needed, depending on units of input)
    r1 = r1 * AU  # Earth's position vector in meters
    r2 = r2 * AU  # Mars' position vector in meters
    
    # Step 1: Compute the chord distance
    c = chord_distance(r1, r2)

    # Step 2: Compute the semi-perimeter
    s = semi_perimeter(r1, r2, c)

    # Step 3: Compute the semi-major axis of the transfer orbit
    a = s / 2

    # Step 4: Calculate the initial guess for x
    # Delta time in seconds (assuming delta_t is given in days)
    delta_t_seconds = delta_t * 86400  # Convert days to seconds
    x_0 = delta_t_seconds / np.sqrt(MU_SUN) * (a**3 / 2) ** 0.5
    
    return x_0

# Example usage:
'''
r1 = np.array([1.0, 0.0, 0.0])  # Earth position in AU (simplified)
r2 = np.array([1.524, 0.0, 0.0])  # Mars position in AU (simplified)
delta_t = 210  # Time of flight in days (example)

initial_guess = izzo_initial_guess(r1, r2, delta_t)
print("Initial guess for Lambert solver (x0):", initial_guess)
'''