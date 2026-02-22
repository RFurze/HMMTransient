import sys
import argparse
import logging
import json
from microscale.src.functions.micro_solver_beta import *
from microscale.src.functions.micro_meshing import GenerateSquareMesh_Scaled

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)    

def load_config(config_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)

def main():
    print(f'currrent directory: {os.getcwd()}')
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run microscale simulation.")
    parser.add_argument("--config", type=str, default="microscale/data/input/input.json", help="Path to configuration file.")
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)
    physical_params = micro_PhysicalParameters(**config["physical_params"])
    solver_settings = micro_SolverSettings(**config["solver_settings"])

    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    input_dir = os.path.join(base_dir, "data/input")
    output_dir = os.path.join(base_dir, "data/output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Define an analytical film thickness for testing
    def ht(x, y, xmax, ymax):
        h = 1e-6
        # h = 1.0e-6 + (0.5e-6)*cos(x * np.pi*8 / xmax) * cos(y * np.pi*8 / ymax)
        return h
    
    def nondim_h(x, y, xmax, ymax, H0, ht=ht):
        return ht(x*xmax, y*ymax, xmax, ymax) / H0


    # Generate mesh
    logger.info("Generating mesh...")
    mesh = GenerateSquareMesh_Scaled(25, 1, 1)

    # Initialize the solver
    logger.info("Initializing the solver...")
    solver = MicroSolver_nondim(mesh, physical_params, solver_settings, k=2, ht=nondim_h)


    target_coordinate = (1, 0)

    # Find the nearest point to the target coordinate
    nearest_point = None
    min_distance = float('inf')

    for vertex in mesh.vertices:
        dist = np.linalg.norm(np.array(vertex.point) - np.array(target_coordinate))
        if dist < min_distance:
            min_distance = dist
            nearest_point = vertex

    if nearest_point is not None:
        print(f'Nearest point to {target_coordinate}: {nearest_point.point}, distance: {min_distance}')
    else:
        print(f'No nearest point found for the target coordinate {target_coordinate}.')

    # Identify the DoF associated with the nearest point
    found = False
    tolerance = 1e-6

    for el in solver.V.Elements():
        for v in el.vertices:
            vertex_coords = mesh[v].point
            if np.allclose(vertex_coords, nearest_point.point, atol=tolerance):
                print('Element with nearest point:', el)
                print('Vertices:', el.vertices)
                print('Global coordinates:', [mesh[v].point for v in el.vertices])
                for dof in el.dofs:
                    print('Associated DoF:', dof)
                found = True
                break
        if found:
            break

    if not found:
        print(f'No element found containing the vertex {nearest_point.nr} with point {nearest_point.point}.')

if __name__ == "__main__":
    main()
