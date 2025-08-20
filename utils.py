import os
from typing import List
import numpy as np
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
import yaml
import sympy as sp
import torch_geometric as pyg
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import to_undirected, is_undirected
from pathlib import Path


def create_data_list(data_dirs: List, seq_len: int, solut_name: str):
    file_list = {}
    it_list_total = []
    case = 0
    
    for i, data_path in enumerate(data_dirs):
        it_list = []
        sim = 0
        for root, dirs, files in os.walk(data_path):
            it_list_sim = []
            file_list[str(sim + case)] = {}
            for name in files:
                if all(x in name for x in [solut_name,'h5']):
                    file_name = os.path.join(root, name)
                    iteration = name.replace(solut_name + '_','')
                    iteration = float(iteration.replace('.h5',''))
                    file_list[str(sim + case)].update({'{}'.format(iteration): file_name})
                    it_list_sim.append((sim + case, sim, iteration))
            sim += 1
            it_list_sim.sort(key=lambda x: x[2])  # Sort by iteration
            it_list.extend(it_list_sim)
        case = sim * (i+1)
    
    # Reshape it_list into sequences of length seq_len
    it_list_total = [it_list[j:j+seq_len] for j in range(0, len(it_list), seq_len) if len(it_list[j:j+seq_len]) == seq_len]
    
    return file_list, it_list_total

def convert_element_to_coo(f):
    # Define the edges for different element types
    element_edges = {
        'hex': [(1, 2), (2, 6), (6, 5), (5, 1), (4, 3), (3, 7), (7, 8), (8, 4), (1, 4), (2, 3), (6, 7), (5, 8)],
        'pri': [(1, 2), (1, 4), (1, 6), (2, 3), (2, 5), (3, 4), (3, 5), (4, 3), (5, 6)],
        'pyr': [(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)],
        'tet': [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    }
    # Loop through the datasets in the file
    rows = []
    cols = []
    cells = []
    for element_type in f.keys():
            ielno = f[element_type]  # Memory-mapped access to HDF5 dataset

            con_list = element_edges[element_type.split('-')[0]]  # Extract the element type

            npel = max((max(con_list)))
            ielno = np.array(ielno).reshape(-1, npel)
            con_list = np.array(con_list)
            
            rows.append(ielno[:,con_list[:,0]-1].reshape(-1) - 1)
            cols.append(ielno[:,con_list[:,1]-1].reshape(-1) - 1)
            cells.append((ielno - 1).tolist())
            
    row = np.concatenate((*rows,*cols))
    col = np.concatenate((*cols,*rows))
    cells = [cell for element_type in cells for cell in element_type]
    edges = np.stack((row, col), axis=0, dtype=np.int32)
    edges = np.sort(edges, axis=0)
    edges = coo_matrix((np.ones(edges[0].shape), (edges[0], edges[1])),shape=((max(row)+1).astype(np.int32),
                                                                              (max(row)+1).astype(np.int32)))
    edges = edges.tocsr()
    edges[edges.nonzero()] = 1
    return edges.tocoo(), cells

def save_edges_to_hdf5(coo_matrix, node_positions, cells, filename='edges.h5'):
    with h5py.File(filename, 'w') as f:
        nonzero_indices = coo_matrix.nonzero()
        f.create_dataset('rows', data=nonzero_indices[0])
        f.create_dataset('cols', data=nonzero_indices[1])
        f.create_dataset('data', data=coo_matrix.data)
        f.create_dataset('node_positions', data=node_positions)
        f.create_dataset('cells', data=cells)
        f.attrs['shape'] = coo_matrix.shape

# Convert mesh to COO and save it to a file (done once)
def convert_and_save_coo(mesh_file, coo_file, coordinate_paths):
    with h5py.File(mesh_file, 'r', driver='core', backing_store=False) as f:
        coord_datasets = []
        
        for axis, path in coordinate_paths.items():
            dataset = find_dataset_by_path(f, path)
            if dataset is None:
                raise ValueError(f"Coordinate {axis} not found at path {path}")
            coord_datasets.append((axis, dataset))
        
        # Check if all coordinate datasets have the same length
        lengths = [dataset.shape[0] for _, dataset in coord_datasets]
        if len(set(lengths)) != 1:
            raise ValueError(f"Coordinate datasets have different lengths: {lengths}")
        
        node_positions = np.column_stack([dataset[:] for _, dataset in coord_datasets])

        # Convert connectivity to COO format
        edge_indices, cells = convert_element_to_coo(f['Connectivity'])
    
    save_edges_to_hdf5(edge_indices, node_positions, cells, coo_file)

def find_dataset_by_path(hdf5_file, path):
    try:
        return hdf5_file[path]
    except KeyError:
        return None

# Memory-map the COO and mesh data
def load_coo_data(coo_file, mesh_file, coordinate_paths):
    if not os.path.exists(coo_file):
        print(f"COO file {coo_file} not found. Generating from mesh file...")
        convert_and_save_coo(mesh_file, coo_file, coordinate_paths)
    
    # Memory-map the COO data (node positions and edge indices)
    with h5py.File(coo_file, 'r') as f:
        node_positions = np.array(f['node_positions'])
        row = f['rows'][:]
        col = f['cols'][:]
        cells = f['cells'][:]
        data = f['data'][:]
        shape = f.attrs['shape']
    return torch.tensor(node_positions, dtype=torch.float), torch.tensor((row, col), dtype=torch.long), cells

# Load simulation data on demand
def load_simulation_data(simulation_file, metadata):
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    variables = {}
    sympy_vars = {}

    with h5py.File(simulation_file, 'r') as f:
        #print(f"HDF5 file structure:")
        #print_hdf5_structure(f)  # Add this helper function

        # Load raw variables
        for variable in metadata['variables']['raw']:
            name = variable['name']
            path = find_path_in_hdf5(f, name)
            if path is None:
                print(f"Warning: Variable '{name}' not found in HDF5 file")
                continue
            variables[name] = f[path][:]
            #print(f"Loaded raw variable '{name}' with shape {variables[name].shape}")

    #print("\nLoaded variables:")
    #for name, value in variables.items():
    #    print(f"{name}: shape {value.shape}, dtype {value.dtype}")

    # Calculate derived variables
    if metadata['variables']['derived']:
        for variable in metadata['variables']['derived']:
            name = variable['name']
            #print(f"\nProcessing derived variable: {name}")
            
            # Create SymPy symbols for all variables used in the expression
            symbols = [sp.Symbol(var) for var in variable['variables']]
            
            # Parse the expression
            expr = sp.sympify(variable['expression'])
            #print(f"Expression: {expr}")
            
            # Create a lambda function from the expression
            lambda_func = sp.lambdify(symbols, expr, modules=['numpy'])
            
            # Evaluate the lambda function with the current variable values
            try:
                args = [variables[var] for var in variable['variables']]
                #print(f"Evaluating {name} with args: {[arg.shape for arg in args]}")
                result = lambda_func(*args)
                variables[name] = result
                #print(f"Successfully calculated {name} with shape {result.shape}")
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                print(f"Variables: {variable['variables']}")
                for var in variable['variables']:
                    if var in variables:
                        print(f"{var} shape: {variables[var].shape}, dtype: {variables[var].dtype}")
                    else:
                        print(f"{var} not found in loaded variables")
                raise

    # Prepare output
    features = []
    for var_name in metadata['variables']['output']:
        if var_name not in variables:
            raise KeyError(f"Output variable '{var_name}' not found in calculated variables.")
        features.append(variables[var_name])

    # Stack all features into a single tensor
    features_tensor = torch.tensor(np.column_stack(features), dtype=torch.float)
    
    return features_tensor

# Create a PyTorch Geometric Data object for a single time step
def create_graph_data(node_positions, edge_indices, simulation_file, metadata, cells):
    node_features = load_simulation_data(simulation_file, metadata)
    # Extract time step from the filename or variable
    if metadata['time']=='filename':
        filename = os.path.basename(simulation_file)
        prefix = metadata['solution_prefix']
        time_step = float(filename[len(prefix)+1:-3])  # Remove prefix, underscore, and '.h5'
        time = time_step * torch.ones(len(node_features), dtype=torch.float)  # Assuming time step is equivalent to time
    else:
         with h5py.File(simulation_file, 'r') as f:
            path = find_path_in_hdf5(f, metadata['time'])
            assert path != None, print(f"Warning: Variable '{metadata['time']}' not found in HDF5 file")
            time_step = f[path][:]
            time = torch.Tensor(time_step) * torch.ones(len(node_features), dtype=torch.float)  # Assuming time step is equivalent to time
    
    # Create a PyTorch Geometric data object
    graph_data = Data(
        target=node_features,         # Node features (velocity, pressure, etc.)
        pos=node_positions,      # Node positions
        edge_index=edge_indices,  # Edges (COO)
        time=time,                # Time
        cells = cells,            # List of cell nodes
        edge_attr = node_positions[edge_indices[1]] - node_positions[edge_indices[0]] #Edge attributes
    )


    # def inspect_target_brief(graph, names=None, nrows=5):
    #     t = graph.target
    #     print(f"target: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    #     if names is not None:
    #         print("channel names:", names)
    #     print("first rows:\n", t[:nrows].detach().cpu())
    # inspect_target_brief(graph_data)

    return graph_data

def find_path_in_hdf5(hdf5_file, target_name):
    def recursive_search(name, item):
        if isinstance(item, h5py.Dataset) and name.split('/')[-1] == target_name:
            return name
        elif isinstance(item, h5py.Group):
            for key, value in item.items():
                result = recursive_search(f"{name}/{key}", value)
                if result:
                    return result
        return None

    return recursive_search('', hdf5_file)

def read_metadata(metadata_file):
    """
    Reads the metadata YAML file and extracts key information.

    Args:
    metadata_file (str): Path to the YAML metadata file.

    Returns:
    dict: A dictionary containing the extracted information.
    """

    # Absolute path to the YAML file and its parent directory
    metadata_path = Path(metadata_file).expanduser().resolve()
    metadata_dir  = metadata_path.parent            # ← base for all relatives

    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Helper: turn *anything* into an absolute Path anchored at metadata_dir
    def to_abs(p):
        p = Path(p).expanduser()
        return (metadata_dir / p).resolve() if not p.is_absolute() else p

    info = {
        "mesh_file"        : str(to_abs(metadata["mesh"]["mesh_file"])),
        "coo_file"         : str(to_abs(metadata["mesh"]["coo_file"])),
        "solution_directory": str(to_abs(metadata["solution"]["folder"])),
        "solution_prefix"  : metadata["solution"]["prefix"],
        "file_format"      : metadata["solution"]["file_format"],
        "dimensions"       : metadata["general"]["dimensions"],
        "solver_name"      : metadata["general"]["solver_name"],
        "seq_len"          : metadata["Dataloader"]["seq_len"],
        "batch_size"       : metadata["Dataloader"]["batch_size"],
        "variables"        : metadata["variables"],
        "time"             : metadata["time"],
        "filter_path"      : str(to_abs(metadata["mesh"]["filter_path"])),
    }
    
    info['coordinate_paths'] = metadata['mesh'].get('coordinate_paths', {'x': 'x', 'y': 'y', 'z': 'z'})

    # ---- validation (unchanged except for Path usage) -----------------------
    if not Path(info["mesh_file"]).is_file():
        raise FileNotFoundError(f"Mesh file not found at {info['mesh_file']}")

    if not Path(info["solution_directory"]).is_dir():
        raise NotADirectoryError(
            f"Solution directory not found at {info['solution_directory']}"
        )

    # Locate newest solution file
    solution_dir = Path(info["solution_directory"])
    solution_files = [
        f for f in solution_dir.iterdir()
        if f.is_file() and f.name.startswith(info["solution_prefix"])
        and f.name.endswith(info["file_format"])
    ]
    if not solution_files:
        raise FileNotFoundError(f"No solution files found in {solution_dir}")

    latest = max(solution_files, key=lambda p: p.stat().st_mtime)
    info["latest_solution"]      = latest.name
    info["latest_solution_path"] = str(latest)

    return info

def print_hdf5_structure(hdf5_file, indent=""):
    for key, item in hdf5_file.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}{key}/")
            print_hdf5_structure(item, indent + "  ")
        else:
            print(f"{indent}{key}: shape {item.shape}, dtype {item.dtype}")

def filter_get_tau_sgs(data, filter, Prandlt = 0.6, momentum_indexes = [1,2,3], rho_index = 0, e_index = 4, p_index = 5, vis_index = 7, T_index = 6):
    #Get Favre filtered velocity fields and tau_sgs
    device = data.x.device
    vis_lam = data.x[:, vis_index].unsqueeze(-1)
    tau_sgs = torch.zeros((data.x.shape[0], 3, 3), device=data.x.device)
    q_sgs = torch.zeros((data.x.shape[0], 3), device=data.x.device)
    p_dilation = torch.zeros((data.x.shape[0]), device=data.x.device)
    #compute fluctuating part of momenta
    u = data.x[:, momentum_indexes] 
    num_graphs = data.batch.max().item() + 1

    data.x[:, 1:] = data.x[:, 1:] * data.x[:, rho_index].unsqueeze(-1)
    u_filtered = filter(data)
    # divide by the filtered density - Favre average
    u_filtered.x[:, 1:] = u_filtered.x[:, 1:] / u_filtered.x[:, rho_index].unsqueeze(-1)
    #Remove density from unfiltered quantities
    data.x[:, 1:] = data.x[:, 1:] / data.x[:, rho_index].unsqueeze(-1)
    filter._set_deterministic(True)
    batch_size = data.num_graphs
    edge_index, edge_weight = pyg.utils.get_laplacian(data.edge_index, normalization='sym')
    batch_laplacian = pyg.utils.to_torch_coo_tensor(edge_index, edge_weight, size=(data.x.size(0), data.x.size(0)))
    
    #Compute SGS stress
    for a, i in enumerate(momentum_indexes):
        for b, j in enumerate(momentum_indexes):
            tau_sgs[:,a,b] = 1/u_filtered.x[:,rho_index] * filter.apply_filter(data.x[:,i] * data.x[:,j] * data.x[:, rho_index], batch_laplacian) - u_filtered.x[:,i] * u_filtered.x[:,j]
    
    #Compute SGS heat flux
    for a, i in enumerate(momentum_indexes):
        q_sgs[:, a] = 1/u_filtered.x[:,rho_index] * filter.apply_filter(data.x[:,i] * data.x[:,e_index] * data.x[:, rho_index], batch_laplacian) - u_filtered.x[:,i] * u_filtered.x[:,e_index]

    #Compute SGS pressure dilation
    div_u = 0.0
    div_u_tilde = 0.0
    u_tilde = u_filtered.x[:, momentum_indexes] 
    #Compute divergences
    grad = LSQGradient(3)
    for a, i in enumerate(momentum_indexes):
        div_u += grad(data.pos, data.x[:, i], data.edge_index)[:, a]
        div_u_tilde += grad(data.pos, u_tilde[:, a], data.edge_index)[:, a]
    p_dilation =  filter.apply_filter(data.x[:,p_index] *  div_u, batch_laplacian) -  u_filtered.x[:,p_index] * div_u_tilde

    #Compute SGS scale scalars
    u_prime = (data.x[:, momentum_indexes] - u_filtered.x[:, momentum_indexes])
    T_prime = (data.x[:, T_index] - u_filtered.x[:, T_index]).unsqueeze(-1)
    grad_u_prime = calc_derivatives(u_prime, data.pos, [0,1,2], data.edge_index)
    grad_T_prime = calc_derivatives(T_prime, data.pos, [0], data.edge_index)
    #print(grad_T_prime.shape)
    S_sgs = torch.stack([
                grad_u_prime[:,0],
                grad_u_prime[:,4],
                grad_u_prime[:,8],
                1/2 * (grad_u_prime[:,1] + grad_u_prime[:,3]),
                1/2 * (grad_u_prime[:,2] + grad_u_prime[:,6]),
                1/2 * (grad_u_prime[:,5] + grad_u_prime[:,7]),
            ], dim = -1)
    for i in range(S_sgs.shape[-1]):
        S_sgs[:, i] = filter.apply_filter(S_sgs[:, i]**2, batch_laplacian)
    eps_sgs = 2 * vis_lam * torch.sum(torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], device=device).view(-1, 6) * S_sgs, dim=-1, keepdim=True)
    q_theta = filter.apply_filter(T_prime[:, 0]**2, batch_laplacian)
    for i in range(grad_T_prime.shape[-1]):
        grad_T_prime[:, i] = filter.apply_filter(grad_T_prime[:, i]**2, batch_laplacian)
    eps_theta = 2 * vis_lam/Prandlt * grad_T_prime.sum(dim=-1, keepdim=True)


    u_filtered.x = torch.cat([u_filtered.x, tau_sgs.reshape(-1,9)], dim=-1)
    u_filtered.x = torch.cat([u_filtered.x, q_sgs], dim=-1)
    u_filtered.x = torch.cat([u_filtered.x, p_dilation.unsqueeze(-1)], dim=-1)
    u_filtered.x = torch.cat([u_filtered.x, eps_sgs], dim=-1)
    u_filtered.x = torch.cat([u_filtered.x, q_theta.unsqueeze(-1)], dim=-1)
    u_filtered.x = torch.cat([u_filtered.x, eps_theta], dim=-1)

    sigma = filter.filter_kwargs['sigma']
    filter._set_deterministic(False)
    return u_filtered, sigma

def calc_derivatives(x, pos, varids, edge_index):

    grad = LSQGradient()
    grads = []
    for i in varids:
        grads.append(grad(pos, x[:,i], edge_index))
    
    return torch.cat(grads, dim=-1)

class FeatureScaler(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.mean = None
        self.var = None
        self.N = None
        self.n_features = n_features
    
    def fit(self, data_list):
        # Stack features from all graphs
        features = torch.cat([data.x[:, :self.n_features] for data in data_list], dim=0)
        self.mean = features.mean(dim=0)
        self.var = features.var(dim=0)
        self.N = torch.sum(torch.tensor([data.x.shape[0] for data in data_list])).item()
        
    @torch.no_grad()
    def forward(self, data):
        device = data.x.device
        with torch.autocast(device_type=data.x.device.type):
            mean = self.mean.to(device)
            std = self.var**0.5
            std = std.to(device)
            # Scale features
            data.x[:, :self.n_features] = (data.x[:, :self.n_features] - mean) / std
            
            # Scale tau_ij components
            #tau_start_idx = 3  # Assuming tau_ij starts after velocity components
            #for i in range(3):
            #    for j in range(3):
            #        idx = tau_start_idx + i*3 + j
            #        data.x[:, idx] = data.x[:, idx] / (std[i] * std[j])
        
        return data

    def update_stats(self, data_list):
        device = data_list[0].x.device
        mean = self.mean.to(device)
        var = self.var.to(device)

        features = torch.cat([data.x[:, :self.n_features] for data in data_list], dim=0)
        new_mean = features.mean(dim=0)
        new_var = features.var(dim=0)
        N_new = torch.sum(torch.tensor([data.x.shape[0] for data in data_list])).item()

        self.mean = self.N/(self.N + N_new) * mean + N_new/(self.N + N_new) * new_mean

        n = self.N + N_new

        self.var = 1/(n-1) * ((N_new-1)*new_var + (self.N-1)*var + (new_mean - mean)**2 * self.N*N_new / n)
        self.N = n


class EdgeScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pos = torch.zeros((1, 3), dtype=torch.bool)
    
    def fit(self, data_list):
        # Find maximum absolute position values across all graphs
        all_pos = torch.cat([data.pos for data in data_list], dim=0)
        self.max_pos = torch.abs(all_pos).max(dim=0)[0]
    
    @torch.no_grad()
    def forward(self, data):
        device = data.edge_attr.device
        max_pos = self.max_pos.to(device)
        with torch.autocast(device_type=data.edge_attr.device.type):
            # Scale edge attributes by maximum position values
            data.edge_attr = data.edge_attr / max_pos
        return data

class LSQGradient(nn.Module):
    def __init__(self, dim: int = 3, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps  # regularization for stability

    def forward(self, pos: torch.Tensor, phi: torch.Tensor, edge_index: torch.Tensor):
        """
        pos: (N, d) node positions
        phi: (N,) scalar field
        edge_index: (2, E) edge list
        Returns: grad_phi (N, d) gradient at each node
        """
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
        row, col = edge_index  # row: target (center), col: neighbor
        dx = pos[col] - pos[row]  # (E, d)
        dphi = phi[col] - phi[row]  # (E,)

        # Weighting: inverse distance squared
        w = 1.0 / (dx.norm(dim=1) + self.eps)**2  # (E,)

        # A_i = sum_j w_ij * dx_ij ⊗ dx_ij
        outer = dx.unsqueeze(2) * dx.unsqueeze(1)  # (E, d, d)
        A = scatter_add(w.view(-1, 1, 1) * outer, row, dim=0, dim_size=pos.size(0))  # (N, d, d)

        # b_i = sum_j w_ij * dphi_ij * dx_ij
        b = scatter_add(w.view(-1, 1) * dphi.unsqueeze(1) * dx, row, dim=0, dim_size=pos.size(0))  # (N, d)

        # Regularize A for inversion
        eye = torch.eye(self.dim, device=pos.device).unsqueeze(0).expand(pos.size(0), -1, -1)
        A_reg = A + self.eps * eye

        # Solve A ∇φ = b
        grad_phi = torch.linalg.solve(A_reg, b.unsqueeze(-1)).squeeze(-1)  # (N, d)

        return grad_phi