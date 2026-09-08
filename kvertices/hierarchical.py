
import numpy as np
import heapq
from .pca_based import alg_2_vertices
from .splitting import split_1d_median, split_1d_kmeans, split_1d_max_gap, split_1d_kde

# Define available splitting methods
SPLITTING_METHODS = {
    "median": split_1d_median,
    "kmeans": split_1d_kmeans,
    "max_gap": split_1d_max_gap,
    "kde": split_1d_kde,
}

def calculate_node_error(X_node, W_node, H_node=None): # H_node often not used/available here
    """Calculates the reconstruction error for a node."""
    if W_node is None or X_node.shape[1] == 0:
        return 0.0 # Or handle appropriately, maybe np.inf if split preferred?
    # Assume W_node is the representative for the cluster (e.g., centroid or vertex)
    # Usually W_node will be shape (m, 1) here
    if W_node.shape[1] == 1:
        # Calculate error w.r.t. the single representative
        recon_error = np.linalg.norm(X_node - W_node, 'fro')**2
    # elif W_node.shape[1] == 2 and H_node is not None:
    #     # If W has 2 cols (from split), H should be (2, n_node)
    #     # This case might be less common for scoring leaf nodes before splitting
    #     recon_error = np.linalg.norm(X_node - W_node @ H_node, 'fro')**2
    else:
        # Fallback: if W has >1 column but no H, use centroid as representative
        centroid = np.mean(X_node, axis=1, keepdims=True)
        recon_error = np.linalg.norm(X_node - centroid, 'fro')**2

    return recon_error

def calculate_node_score_error(node_data):
    """Calculates score based on reconstruction error."""
    return node_data['error']

def calculate_node_score_size_weighted_error(node_data):
    """Calculates score based on size-weighted reconstruction error."""
    return node_data['error'] * node_data['size']

# Define available splitting criteria scorers
SPLITTING_CRITERIA = {
    "max_error": calculate_node_score_error,
    "max_size_weighted_error": calculate_node_score_size_weighted_error,
    # Add other criteria like Min Vol Potential if implemented
}

def hierarchical_k_vertices(
    X: np.ndarray,
    k_target: int,
    splitting_criterion: str = "max_size_weighted_error",
    splitting_method: str = "median",
    final_H_method: str = 'slsqp', # Method to compute final H ('slsqp' or 'nnls')
    alpha_H_final: float = None, # Parameter if final_H_method='nnls'
    nnls_solver_final: str = 'blockpivot', # Parameter if final_H_method='nnls'
    max_levels: int = None, # Optional limit on hierarchy depth
    min_cluster_size: int = 1, # Minimum points to allow a split
    verbose: bool = False
) -> (np.ndarray, np.ndarray, list):
    """
    Implements the Hierarchical K-Vertices algorithm (Algorithm 3).

    Constructs a k-cluster solution by recursively applying the 2-vertices
    splitting procedure.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        Data matrix (features x samples).
    k_target : int
        Target number of leaf clusters.
    splitting_criterion : str, optional (default="max_size_weighted_error")
        Criterion to select the next node to split. Options:
        ["max_error", "max_size_weighted_error"].
    splitting_method : str, optional (default="median")
        Method used for 1D thresholding after alg_2_vertices. Options:
        ["median", "kmeans", "max_gap", "kde"].
    final_H_method : str, optional (default='slsqp')
        Method used to compute the final H matrix based on the hierarchically
        derived W_final. Options: 'slsqp', 'nnls'.
    alpha_H_final : float, optional
        Penalty parameter for the soft sum-to-one constraint when final_H_method='nnls'.
        If None, a default value is used.
    nnls_solver_final : str, optional (default='blockpivot')
        NNLS solver to use when final_H_method='nnls'.
    max_levels : int, optional
        Maximum depth of the hierarchy tree. If None, stops when k_target reached.
    min_cluster_size : int, optional (default=1)
        Minimum number of points required in a cluster to consider splitting it.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    W_final : np.ndarray, shape (m, k_actual)
        The final matrix of k_actual vertices (representatives of leaf clusters).
        k_actual might be different from k_target if splitting stops early.
    H_final : np.ndarray, shape (k_actual, n)
        The final membership matrix (computed using W_final and full X via
        `final_H_method`).
    leaf_nodes : list
        List containing information about the final k_actual leaf nodes.
    """
    m, n = X.shape
    if k_target < 1:
        raise ValueError("k_target must be at least 1")
    if k_target == 1:
        # Handle k=1 case: single cluster (centroid)
        W_final = np.mean(X, axis=1, keepdims=True)
        # Compute H using the chosen method (trivial for k=1)
        H_final = np.ones((1, n))
        leaf_nodes = [{'id': 0, 'indices': np.arange(n), 'W': W_final, 'level': 0, 'size': n, 'error': calculate_node_error(X, W_final)}]
        return W_final, leaf_nodes
    if k_target > n:
        print(f"Warning: k_target ({k_target}) > n_samples ({n}). Setting k_target=n.")
        k_target = n

    # Validate inputs
    if splitting_criterion not in SPLITTING_CRITERIA:
        raise ValueError(f"Unknown splitting_criterion: {splitting_criterion}. Available: {list(SPLITTING_CRITERIA.keys())}")
    if splitting_method not in SPLITTING_METHODS:
        raise ValueError(f"Unknown splitting_method: {splitting_method}. Available: {list(SPLITTING_METHODS.keys())}")
    if final_H_method not in ['slsqp', 'nnls']:
        raise ValueError(f"Unknown final_H_method: {final_H_method}. Choose 'slsqp' or 'nnls'.")


    split_scorer = SPLITTING_CRITERIA[splitting_criterion]
    split_function = SPLITTING_METHODS[splitting_method]

    # Initialize
    node_counter = 0
    root_indices = np.arange(n)
    root_W = np.mean(X, axis=1, keepdims=True) # Initial W is centroid
    root_node = {
        'id': node_counter,
        'indices': root_indices,
        'W': root_W,
        # 'H': np.ones((1, n)), # H relative to node is less useful here
        'error': calculate_node_error(X, root_W),
        'size': n,
        'level': 0
    }
    node_counter += 1

    # Priority queue stores (-score, node_id, node_dict)
    # Use negative score because heapq is a min-heap
    initial_score = split_scorer(root_node)
    priority_queue = [(-initial_score, root_node['id'], root_node)]
    heapq.heapify(priority_queue)

    active_leaves = {root_node['id']: root_node}
    num_leaves = 1
    # current_level = 0 # Tracked per node

    while num_leaves < k_target:
        if not priority_queue:
            if verbose:
                print("Priority queue is empty, cannot split further.")
            break # Cannot split further

        # Select node with highest score to split
        # Use loop to handle cases where top node was already processed or invalid
        node_to_split = None
        while priority_queue and node_to_split is None:
             neg_score, node_id_to_split, potential_node = heapq.heappop(priority_queue)

             # Check if node still exists in active_leaves (hasn't been replaced)
             if node_id_to_split not in active_leaves:
                 continue # Node already split/removed

             # Check if node is same as the one popped (heapq stores dict copies?)
             # This check might be redundant if IDs are unique and active_leaves is correct
             if potential_node['id'] != active_leaves[node_id_to_split]['id']:
                  continue

             node_to_split = active_leaves[node_id_to_split]

             # Check size constraint
             if node_to_split['size'] < max(min_cluster_size * 2, 2): # Need at least 2 points to split
                 if verbose:
                     print(f"Node {node_id_to_split} too small ({node_to_split['size']}). Skipping split.")
                 node_to_split = None # Mark as invalid, continue loop
                 continue

             # Check level constraint
             current_level = node_to_split['level']
             if max_levels is not None and current_level >= max_levels:
                 if verbose:
                     print(f"Node {node_id_to_split} reached max level {max_levels}. Skipping split.")
                 node_to_split = None # Mark as invalid, continue loop
                 continue


        if node_to_split is None:
             # If loop finishes without finding a valid node, break
             if verbose:
                 print("No suitable nodes left to split in the priority queue.")
             break

        # We have a valid node_to_split
        node_id_to_split = node_to_split['id']
        current_indices = node_to_split['indices']
        X_current = X[:, current_indices]
        current_level = node_to_split['level']

        # --- Perform the split ---
        try:
            W_split, H_split, eta = alg_2_vertices(X_current)
        except Exception as e:
            if verbose:
                print(f"Error during alg_2_vertices for node {node_id_to_split}: {e}. Skipping split.")
            # Remove node from active leaves to prevent re-selection?
            # Or just leave it - won't be selected again if error persists
            # del active_leaves[node_id_to_split] # Maybe safer?
            continue # Skip split if 2-vertices fails

        # Check if eta is degenerate (all points are the same in the 1D projection)
        if X_current.shape[1] > 1 and np.abs(np.min(eta) - np.max(eta)) < 1e-9:
            if verbose:
                print(f"Node {node_id_to_split} has degenerate eta (variance ~0). Skipping split.")
            # Remove from active leaves to prevent infinite loop if always degenerate
            # del active_leaves[node_id_to_split]
            continue # Cannot split based on eta

        try:
            # Use the chosen splitting method to get indices for the two children
            # The splitting function should return two lists/arrays of *local* indices (relative to X_current)
            local_indices1, local_indices2 = split_function(eta)
        except Exception as e:
            if verbose:
                print(f"Error during splitting method '{splitting_method}' for node {node_id_to_split}: {e}. Skipping split.")
            continue

        # Ensure split is valid (non-empty children meeting min size)
        if len(local_indices1) < min_cluster_size or len(local_indices2) < min_cluster_size:
            if verbose:
                print(f"Node {node_id_to_split} split resulted in child smaller than min_cluster_size ({min_cluster_size}). Skipping split.")
            # Put the parent back? No, just don't split it. Leave it as a leaf.
            continue
        # ------------------------

        if verbose:
            print(f"Splitting node {node_id_to_split} (size {node_to_split['size']}) at level {current_level} -> {len(local_indices1)}, {len(local_indices2)} points.")

        # Remove parent node from active leaves
        del active_leaves[node_id_to_split]

        # Create child nodes
        global_indices1 = current_indices[local_indices1]
        global_indices2 = current_indices[local_indices2]
        X_child1 = X[:, global_indices1]
        X_child2 = X[:, global_indices2]

        # Use the vertices from alg_2_vertices as representatives for the children
        W_child1 = W_split[:, [0]] # First vertex for child 1
        W_child2 = W_split[:, [1]] # Second vertex for child 2

        child1_id = node_counter
        node_counter += 1
        child1_node = {
            'id': child1_id,
            'indices': global_indices1,
            'W': W_child1,
            # 'H': H_split[:, local_indices1], # H relative to child - not needed after split
            'error': calculate_node_error(X_child1, W_child1), # Error uses representative W
            'size': len(global_indices1),
            'level': current_level + 1
        }
        # Only add to queue if size allows further splitting
        if child1_node['size'] >= max(min_cluster_size * 2, 2):
            score1 = split_scorer(child1_node)
            heapq.heappush(priority_queue, (-score1, child1_id, child1_node))
        active_leaves[child1_id] = child1_node


        child2_id = node_counter
        node_counter += 1
        child2_node = {
            'id': child2_id,
            'indices': global_indices2,
            'W': W_child2,
            # 'H': H_split[:, local_indices2],
            'error': calculate_node_error(X_child2, W_child2),
            'size': len(global_indices2),
            'level': current_level + 1
        }
        # Only add to queue if size allows further splitting
        if child2_node['size'] >= max(min_cluster_size * 2, 2):
             score2 = split_scorer(child2_node)
             heapq.heappush(priority_queue, (-score2, child2_id, child2_node))
        active_leaves[child2_id] = child2_node

        num_leaves += 1 # Increase leaf count by 1 (split removed 1, added 2)

    # --- Post-processing: Collect results from leaf nodes ---
    leaf_nodes = list(active_leaves.values())
    # Sort leaves by ID for consistency (optional)
    leaf_nodes.sort(key=lambda node: node['id'])

    k_actual = len(leaf_nodes)
    if k_actual == 0: # Should not happen if n > 0
         print("Warning: No leaf nodes found. Returning empty results.")
         return np.zeros((m, 0)), np.zeros((0, n)), []

    if k_actual != k_target and verbose:
        print(f"Warning: Final number of leaves ({k_actual}) differs from k_target ({k_target}) due to splitting constraints.")

    # Collect final W matrix (representatives from leaves)
    W_final = np.hstack([node['W'] for node in leaf_nodes])

    return W_final, leaf_nodes