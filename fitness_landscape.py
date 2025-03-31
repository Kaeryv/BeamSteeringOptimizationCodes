import numpy as np
import sys
sys.path.append("./user/")

def random_orthogonal_unit_vectors(N):
    """
    Generate two random, orthogonal unit vectors in N-dimensional space.
    
    Parameters:
    N (int): Dimension of the space.
    
    Returns:
    tuple: Two N-dimensional unit vectors that are orthogonal.
    """
    # Generate a random unit vector
    v1 = np.random.randn(N)
    v1 /= np.linalg.norm(v1)  # Normalize to unit length
    
    # Generate another random vector
    v2 = np.random.randn(N)
    
    # Remove the component of v2 that is in the direction of v1
    v2 -= np.dot(v2, v1) * v1  
    
    # Normalize to unit length
    v2 /= np.linalg.norm(v2)
    
    return v1, v2

def generate_meshgrid(v1, v2, min_vals, max_vals, num_points=50):
    """
    Generate a mesh grid spanning the basis of two orthogonal vectors.
    
    Parameters:
    v1, v2 (ndarray): Two orthogonal unit vectors.
    min_vals, max_vals (ndarray): Min and max bounds for each dimension.
    num_points (int): Number of points along each axis.
    
    Returns:
    ndarray: A set of grid points in the original N-dimensional space.
    """
    t1 = np.linspace(-1, 1, num_points)
    t2 = np.linspace(-1, 1, num_points)
    T1, T2 = np.meshgrid(t1, t2)
    
    # Create the grid in the subspace spanned by v1 and v2
    grid_points = np.array([t1 * v1 + t2 * v2 for t1, t2 in zip(T1.ravel(), T2.ravel())])
    
    # Rescale each dimension based on min_vals and max_vals
    grid_points = grid_points * (max_vals - min_vals) / 2 + (max_vals + min_vals) / 2
    
    return grid_points.reshape(num_points, num_points, -1)

from types import SimpleNamespace
Nell = 8
sim_args = SimpleNamespace(
    elow=2.0,
    ehigh=4.0,
    wavelength=1.01,
    bilayer_mode="free",
    num_layers=16,
    pw=(int(sys.argv[2]), 1),
    polarizations=["X"],#, "Y", "XY", "LCP", "RCP"],  # LCP+ RCP-
    angles=[0, 60, 31],
    parameterization="ellipsis",
    target_order=(-1, +1),
    #parameterization_args={"num_layers": 14, "harmonics": [0.5,1,1.5]},
    #parameterization_args={"num_items": 12, "num_layers": 16,"depth": 3.2},#,"materials": [2.0, 3.0, 4.0 ]
    parameterization_args={"num_items": Nell, "num_layers": 16,"depth": 3.2, "materials": [2.0, 3.0, 4.0 ]},#,
)   
filename = sys.argv[3]
if __name__ == "__main__":
    action = sys.argv[1]
    if action == "c":
        np.random.seed(int(sys.argv[4]))
        from user.multigrating import __run__ as sim
        # Example usage
        N = Nell*5  # Change this to any desired dimension
        v1, v2 = random_orthogonal_unit_vectors(N)
        min_vals = 2*Nell * [-0.5]  # Example min values per dimension
        min_vals.extend(2*Nell * [0.0])
        min_vals.extend(Nell * [-0.7854])
        max_vals = 2*Nell * [0.5]
        max_vals.extend(2*Nell * [0.5])
        max_vals.extend(Nell * [0.7854])

        min_vals = np.array(min_vals)
        max_vals = np.array(max_vals)
        num_points = 128
        grid = generate_meshgrid(v1, v2, min_vals, max_vals, num_points=num_points)
        print("Meshgrid shape:", grid.shape)
        Ndim = Nell * 5
        fitness = []
        for design in grid.reshape(-1, Ndim):
            fitness.append(sim(None,  design, sim_args, figpath=None)["fitness"])
        np.savez(filename, fitness=fitness, num_points=num_points)

    elif action == "p":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        d = np.load(filename)
        N = d["num_points"]
        ax.matshow(d["fitness"].reshape(N,N))
        fig.savefig("test.png")
