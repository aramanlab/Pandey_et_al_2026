# Date  : 28th May 2026
# Author: Bipul Pandey

import os
import copy
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor


class PCA_Tree_Builder():
    """
    Builds a phylogenetic-style tree from a reference dataset using PCA-based
    distances. New (generated) sequences can be projected into the reference
    PCA space and placed onto the reference tree.

    Parameters
    ----------
    Compact_Reference_Dataset : np.ndarray, shape (N, L)
        Reference sequences in compact (non-one-hot) form.
    mode : str
        Distance weighting mode. One of:
            "variance" — multiply by explained variance ratio (amplifies major PCs)
            "inverse"  — multiply by inverse variance (amplifies minor PCs)
            "none"     — plain Euclidean distance in PC space
    Tree_method : str
        Tree construction algorithm. One of "nj" (Neighbor-Joining) or "upgma".
    Use_only_unique : bool
        If True, PCA is fit on unique sequences only. Distances are still
        computed on the full dataset.
    Alphabets : array-like or None
        Explicit alphabet. If None, inferred from reference data.
        If supplied, the union with the reference alphabet is used.
    """

    def __init__(self,
                 Compact_Reference_Dataset,
                 mode="variance",
                 Tree_method="nj",
                 Use_only_unique=False,
                 Alphabets=None):

        # ── store raw data ────────────────────────────────────────────────────
        self.Compact_Reference_Data = Compact_Reference_Dataset
        self.num_raw_features       = np.shape(self.Compact_Reference_Data)[1]
        self.distance_mode          = mode
        self.tree_building_method   = Tree_method
        self.use_only_unique        = Use_only_unique

        # ── resolve alphabet ──────────────────────────────────────────────────
        self.Alphabets = self._resolve_alphabets(Alphabets)

        # ── one-hot encode full reference data ───────────────────────────────
        self.Unfurled_Reference_Data = self._one_hot_encode_array(self.Compact_Reference_Data)
        # ── optionally use only unique sequences for PCA fit ─────────────────
        if self.use_only_unique:
            print("Use Unique = True")
            print("PCA will be fit on UNIQUE datapoints only.")
            unique_compact = np.unique(self.Compact_Reference_Data, axis=0)
            self.Unfurled_Reference_Data_Used = self._one_hot_encode_array(unique_compact)
        else:
            print("Use Unique = False (default)")
            print("PCA will be fit on ALL datapoints.")
            self.Unfurled_Reference_Data_Used = self.Unfurled_Reference_Data

        # ── dimensions ───────────────────────────────────────────────────────
        # PCA components limited by the data used for fitting
        n_fit, f_fit             = np.shape(self.Unfurled_Reference_Data_Used)
        self.num_pca_components  = int(np.min((n_fit, f_fit)))

        # bookkeeping on full dataset
        self.num_datapoints, self.num_unfurled_features = np.shape(self.Unfurled_Reference_Data)

        # ── readable labels ───────────────────────────────────────────────────
        self.text_tree_build = "UPGMA" if self.tree_building_method == "upgma" \
                               else "Neighbor-Joining"

        # ── print summary ─────────────────────────────────────────────────────
        print("|| --------------------------------------------------------- ||")
        print(f"|| Number of Alphabets      = {len(self.Alphabets)}")
        print(f"|| Number of Data Points    = {self.num_datapoints}")
        if self.use_only_unique:
            print(f"|| Number of Unique Points  = {len(unique_compact)}")
        print(f"|| Number of Features       = {self.num_unfurled_features}")
        print(f"|| Number of PCA Components = {self.num_pca_components}")
        print("|| --------------------------------------------------------- ||")
        print(f"|| Distance Metric = {mode} weighted Euclidean Distance")
        print(f"|| Tree Building   = {self.text_tree_build}")
        print("|| --------------------------------------------------------- ||")
        print(" ----- Performing PCA ----- ")
        self.pca = PCA(n_components=self.num_pca_components)
        self.pca.fit(self.Unfurled_Reference_Data_Used)
        print(" -----  PCA Complete  ----- ")
        print("If settings are correct, call Reference_Tree_Construction().")

    # ═════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════════════

    def Reference_Tree_Construction(self):
        """
        Fit PCA on the reference data, compute pairwise distances,
        and build the reference tree. Must be called before New_Data_To_Tree.

        Returns
        -------
        ref_tree : Bio.Phylo tree object
        """
        # ── PCA Projection ───────────────────────────────────────────────────────────────
        self.Reference_Data_Projected = self.pca.transform(self.Unfurled_Reference_Data)
        # ── distance weights ──────────────────────────────────────────────────
        lam = self.pca.explained_variance_ratio_
        if self.distance_mode == "variance":
            print("Weighting PCA projection by Explained Variance Ratio.")
            self.distance_weight = lam + 1e-10
        elif self.distance_mode == "inverse":
            print("Weighting PCA projection by Inverse Explained Variance Ratio.")
            self.distance_weight = 1.0 / (lam + 1e-10)
        else:
            print("No weighting — plain Euclidean distance in PC space.")
            self.distance_weight = np.ones(self.num_pca_components)
        # ── global centroid (computed once, used for all placements) ──────────
        self.Reference_Centroid = np.mean(self.Reference_Data_Projected, axis=0)

        # ── pairwise distance matrix ──────────────────────────────────────────
        print("Constructing pairwise distance matrix...")
        self.Reference_Distance_Matrix, self.Reference_Distance_Matrix_Phylo = \
            self._pairwise_distance_matrix(
                self.Reference_Data_Projected,
                self.distance_weight,
                Denoter="Ref")
        self.Reference_Data_name = self.Reference_Distance_Matrix_Phylo.names
        print("Distance matrix constructed.")

        # ── tree ──────────────────────────────────────────────────────────────
        print(f"Constructing reference tree using {self.text_tree_build}...")
        self.ref_tree = self._tree_construction(
            self.Reference_Distance_Matrix_Phylo,
            method=self.tree_building_method)
        print("Reference tree constructed.")
        return self.ref_tree
    

    def Reference_Tree_Construction(self):
        """
        Fit PCA on the reference data, compute pairwise distances,
        and build the reference tree. Must be called before New_Data_To_Tree.

        Returns
        -------
        ref_tree : Bio.Phylo tree object
        """
        # ── PCA ───────────────────────────────────────────────────────────────
        self.pca = PCA(n_components=self.num_pca_components)
        self.pca.fit(self.Unfurled_Reference_Data_Used)
        self.Reference_Data_Projected = self.pca.transform(self.Unfurled_Reference_Data)

        # ── distance weights ──────────────────────────────────────────────────
        lam = self.pca.explained_variance_ratio_
        if self.distance_mode == "variance":
            print("Weighting PCA projection by Explained Variance Ratio.")
            self.distance_weight = lam + 1e-10
        elif self.distance_mode == "inverse":
            print("Weighting PCA projection by Inverse Explained Variance Ratio.")
            self.distance_weight = 1.0 / (lam + 1e-10)
        else:
            print("No weighting — plain Euclidean distance in PC space.")
            self.distance_weight = np.ones(self.num_pca_components)

        # ── global centroid (computed once, used for all placements) ──────────
        self.Reference_Centroid = np.mean(self.Reference_Data_Projected, axis=0)

        # ── pairwise distance matrix ──────────────────────────────────────────
        print("Constructing pairwise distance matrix...")
        self.Reference_Distance_Matrix, self.Reference_Distance_Matrix_Phylo = \
            self._pairwise_distance_matrix(
                self.Reference_Data_Projected,
                self.distance_weight,
                Denoter="Ref")
        self.Reference_Data_name = self.Reference_Distance_Matrix_Phylo.names
        print("Distance matrix constructed.")

        # ── tree ──────────────────────────────────────────────────────────────
        print(f"Constructing reference tree using {self.text_tree_build}...")
        self.ref_tree = self._tree_construction(
            self.Reference_Distance_Matrix_Phylo,
            method=self.tree_building_method)
        print("Reference tree constructed.")
        return self.ref_tree




    def New_Data_To_Tree(self, New_Compact_Data, Denoter="Y", placement_option="B"):
        """
        Project new (generated) sequences into the reference PCA space,
        find their nearest reference neighbor, and place them on the tree.

        Parameters
        ----------
        New_Compact_Data : np.ndarray, shape (M, L)
            New sequences in compact form.
        Denoter : str
            Prefix for naming new sequences (e.g. "Gen").

        placement_option : str
            "A" — attach new leaf directly from parent of NN with the same branch length as NN.          
            "B" — split the NN edge at the geometrically estimated position with geometric branch length.
            "C" — attach new leaf directly from parent of NN with geometric branch length.
        
        Notes on Options:
        placement_option "A" - natural for UPGMA
        placement_option "B" - natural for neighbor-joining and more consistent globally 
        placement_option "C" - natural for neighbor-joining but less consistent globally 

        Returns
        -------
        Placement_nn  : dict  — placement info per new sequence
        Placed_Tree   : Bio.Phylo tree with new leaves inserted
        """
        # Step 1: project to PCA space
        New_Data_names, New_Data_Projected = self._project_to_pca(
            New_Compact_Data, Denoter)

        # Step 2: find nearest reference leaf + compute branch geometry
        Placement_nn = self._find_nearest_leaf(
            self.Reference_Data_Projected,
            self.Reference_Data_name,
            New_Data_Projected,
            New_Data_names)

        # Step 3: place onto tree
        Placed_Tree = self._place_on_tree(self.ref_tree, Placement_nn, placement_option=placement_option)
        
        # Step 4: Suggest better options
        #-----
        if self.tree_building_method =="upgma" and placement_option !="A":
            print("Note: Placement Option A is more natural for UPGMA")

        elif self.tree_building_method =="nj" and placement_option =="A":
            print("Note: Placement Option B is more natural for Neighbor Joining")
        #-----

        return Placement_nn, Placed_Tree

    def Save_Tree_To_Newick(self, Tree, filename="Tree", file_loc=None):
        """
        Save a Bio.Phylo tree to a Newick file.

        Parameters
        ----------
        Tree      : Bio.Phylo tree object
        filename  : str   base filename (no extension)
        file_loc  : str or None   directory path; uses cwd if None
        """
        f_name    = filename + "_" + self.tree_building_method + ".nwk"
        full_path = f_name if file_loc is None else os.path.join(file_loc, f_name)
        Phylo.write(Tree, full_path, "newick")
        print(f"Tree saved to {full_path}")
        return full_path

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE — ALPHABET & ENCODING
    # ═════════════════════════════════════════════════════════════════════════

    def _resolve_alphabets(self, Alphabets):
        """Determine the alphabet from reference data and optional user input."""
        Alphabets_from_ref = set(np.unique(self.Compact_Reference_Data))

        if Alphabets is None:
            Alphabets_union = Alphabets_from_ref
        else:
            Alphabets_supplied   = set(Alphabets)
            Symmetric_difference = Alphabets_from_ref ^ Alphabets_supplied

            if Symmetric_difference == set():
                Alphabets_union = Alphabets_supplied
            else:
                print("Supplied alphabet differs from reference alphabet.")
                print("Using union of both as the working alphabet.")
                Alphabets_union = Alphabets_supplied | Alphabets_from_ref

        return np.array(sorted(Alphabets_union))

    def _one_hot_encode_array_vectorized(self, Array):
        """
        Vectorised one-hot encoding.
        Array : (N, L) → output : (N, L * q)
        """
        # (N, L, q) boolean tensor — True where Array[n,l] == Alphabet[q]
        matches = (Array[:, :, None] == self.Alphabets[None, None, :])
        return matches.reshape(Array.shape[0], -1).astype(np.float32)
    
    def _one_hot_encode_sequence(self, Sequence):
        """
         One-hot encoding of a given sequence.
        """
        ohe = []
        for elem in Sequence:
            elem_ohe      = np.zeros(len(self.Alphabets))
            arg           = np.argwhere(self.Alphabets==elem).reshape(-1)[0]
            elem_ohe[arg] =1
            ohe.append(elem_ohe)
        ohe = np.concatenate(ohe)
        return(ohe)
    
    def _one_hot_encode_array(self, Array):
        """
         One-hot encoding of a given array.
        Array : (N, L) → output : (N, L * q)
        """
        Array_ohe = []
        for seq in Array:
            seq_ohe = self._one_hot_encode_sequence(seq)
            Array_ohe.append(seq_ohe)
        Array_ohe = np.array(Array_ohe)

        return Array_ohe

    def _data_names(self, Num_Data, Denoter="X"):
        """Generate sequence names: Denoter-0, Denoter-1, ..."""
        return [f"{Denoter}-{n}" for n in range(Num_Data)]

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE — DISTANCES & TREE
    # ═════════════════════════════════════════════════════════════════════════

    def _pairwise_distance_matrix(self, Z, weights, Denoter="X"):
        """
        Compute pairwise weighted Euclidean distances via scipy pdist.
        seuclidean divides by V, so we pass V = 1/weights to get multiply-by-weights.

        Returns
        -------
        Distance_Matrix       : np.ndarray (N, N)
        Distance_Matrix_Phylo : Bio.Phylo DistanceMatrix (lower triangle)
        """
        condensed       = pdist(Z, metric="seuclidean", V=1.0 / weights)
        Distance_Matrix = squareform(condensed)

        # BioPython needs lower triangle as list of lists
        lower_triangle = [
            list(Distance_Matrix[i, :i + 1])
            for i in range(Distance_Matrix.shape[0])]
        
        Names                 = self._data_names(Distance_Matrix.shape[0], Denoter)
        Distance_Matrix_Phylo = DistanceMatrix(names=Names, matrix=lower_triangle)

        return Distance_Matrix, Distance_Matrix_Phylo

    def _tree_construction(self, Distance_matrix_lower_triangle, method):
        """Build NJ or UPGMA tree from a BioPython DistanceMatrix."""
        constructor = DistanceTreeConstructor()
        if method == "nj":
            return constructor.nj(Distance_matrix_lower_triangle)
        elif method == "upgma":
            return constructor.upgma(Distance_matrix_lower_triangle)
        else:
            raise ValueError(f"Unknown tree method '{method}'. Choose 'nj' or 'upgma'.")

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE — PCA PROJECTION & NEAREST LEAF
    # ═════════════════════════════════════════════════════════════════════════

    def _project_to_pca(self, New_Compact_Data, Denoter="Y"):
        """
        Validate and project new sequences into the reference PCA space.

        Raises
        ------
        TypeError  : if sequence length differs from reference
        ValueError : if new data contains symbols outside the reference alphabet
        """
        num_new_data, num_new_raw_features = np.shape(New_Compact_Data)
        # idiot-proofing
        # 1. shape check first (O(1))
        if num_new_raw_features != self.num_raw_features:
            raise TypeError(
                f"New data has {num_new_raw_features} features; "
                f"reference has {self.num_raw_features}.")

        # 2. alphabet check (O(N*L))
        diff_alphabet = set(np.unique(New_Compact_Data)) - set(self.Alphabets)
        if diff_alphabet:
            raise ValueError(
                f"New data contains unseen symbol(s): {diff_alphabet}")

        Unfurled_new_data = self._one_hot_encode_array(New_Compact_Data)
        New_Data_Projected = self.pca.transform(Unfurled_new_data)
        New_Data_names     = self._data_names(num_new_data, Denoter)

        return New_Data_names, New_Data_Projected

    def _weighted_distance(self, u, v):
        """Weighted Euclidean distance between two PC vectors."""
        diff = u - v
        return float(np.sqrt(np.sum(self.distance_weight * diff ** 2)))

    def _compute_placement_geometry(self, d_direct, d_nn, d_new):
        """
        Use the law of cosines on the triangle:

              Global Centroid
               /           \
             d_nn         d_new
             /               \
            NN ── d_direct ── New Leaf

        to find:
            branch_length : perpendicular distance from internal node to new leaf
            split_position: distance along NN→parent edge where the split occurs

        Parameters
        ----------
        d_direct : float   distance between NN and new leaf
        d_nn     : float   distance from NN to global centroid
        d_new    : float   distance from new leaf to global centroid

        Returns
        -------
        branch_length  : float
        split_position : float
        """
        
        # Step 1: we get angle at NN via law of cosines
        # d_new² = d_nn² + d_direct² - 2*d_nn*d_direct*cos(θ_nn)
        # angle at NN via law of cosines
        denom    = 2.0 * d_nn * d_direct + 1e-10
        cos_theta = np.clip((d_nn**2 + d_direct**2 - d_new**2) / denom, -1.0, 1.0)
        theta     = np.arccos(cos_theta)
        
        # Step 2: we then get the perpendicular and parallel components of d_direct

        # perpendicular component -> branch length of new leaf
        # this gives the branch length
        branch_length  = float(d_direct * np.sin(theta))

        # projection along NN -> centroid axis -> split position along NN edge
        # this gives the position where a new intermediate node is added
        split_position = float(d_direct * np.cos(theta))

        return branch_length, split_position

    def _find_nearest_leaf(self, Ref_Data_Projected, Ref_Data_Name,
                           New_Data_Projected, New_Data_Name):
        """
        For each new sequence find its nearest reference neighbor in weighted
        PC space, then compute placement geometry using the global centroid.

        Returns
        -------
        Placement_nn : dict
            Keys are new sequence names. Values are dicts with:
                nn_ref_name        : name of nearest reference sequence
                nn_ref_arg         : index of nearest reference sequence
                nn_ref_dist        : weighted PC distance to NN
                elem_pca_coord     : PC coordinates of new sequence
                nn_pca_coord       : PC coordinates of NN
                elem_centroid_dist : distance from new seq to global centroid
                nn_centroid_dist   : distance from NN to global centroid
                branch_length      : estimated branch length (from geometry)
                split_position     : estimated split along NN edge (Option B)
        """
        Placement_nn = {}

        for i, elem in enumerate(New_Data_Projected):
            elem_name = New_Data_Name[i]

            # ── nearest neighbor ──────────────────────────────────────────────
            Diff      = Ref_Data_Projected - elem                      # (N_ref, k)
            Distances = np.sqrt(np.sum(self.distance_weight * Diff**2, axis=1))
            nn_arg    = int(np.argmin(Distances))
            d_direct  = float(Distances[nn_arg])
            nn_coord  = Ref_Data_Projected[nn_arg]
            nn_name   = Ref_Data_Name[nn_arg]

            # ── centroid distances ────────────────────────────────────────────
            d_new = self._weighted_distance(elem,    self.Reference_Centroid)
            d_nn  = self._weighted_distance(nn_coord, self.Reference_Centroid)

            # ── law of cosines placement geometry ────────────────────────────
            branch_length, split_position = self._compute_placement_geometry(d_direct, d_nn, d_new)

            Placement_nn[elem_name] = {
                "nn_ref_name":        nn_name,
                "nn_ref_arg":         nn_arg,
                "nn_ref_dist":        d_direct,
                "elem_pca_coord":     elem,
                "nn_pca_coord":       nn_coord,
                "elem_centroid_dist": d_new,
                "nn_centroid_dist":   d_nn,
                "branch_length":      branch_length,
                "split_position":     split_position,
            }
        return Placement_nn

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE — TREE PLACEMENT
    # ═════════════════════════════════════════════════════════════════════════

    def _find_parent(self, tree, leaf_name):
        """
        Return the parent Clade of a named leaf in the given tree.
        Falls back to tree root if the leaf is a direct child of root.
        """
        path = tree.get_path(leaf_name)
        if len(path) < 2:
            return tree.root
        return path[-2]

    def _place_on_tree(self, Tree, Placement_nn, placement_option="B"):
        """
        Insert new leaves into a deep copy of the reference tree.

        placement_option = "A":
            New leaf branches directly from the parent of NN.
            Branch length = length as NN from the parent.
            This is natural for UPGMA.


        placement_option = "B":
            The NN edge is split at split_position from NN.
            An intermediate node is inserted; new leaf hangs off it.
            Branch length = geometrically estimated perpendicular distance.


        placement_option = "C":
            New leaf branches directly from the parent of NN.
            Branch length = geometrically estimated perpendicular distance.
            Simpler Method. Minimal change to the Ref Tree

        Parameters
        ----------
        Tree             : Bio.Phylo tree (reference tree, not modified)
        Placement_nn     : dict from _find_nearest_leaf
        placement_option : "C" or "B"

        Returns
        -------
        Placed_Tree : Bio.Phylo tree with new leaves inserted
        """
        Placed_Tree = copy.deepcopy(Tree)

        for elem, info in Placement_nn.items():
            nn_name       = info["nn_ref_name"]
            branch_length = info["branch_length"]
            split_pos     = info["split_position"]

            try:
                if placement_option == "A":
                    # ── Option A: mirror NN branch length from parent ─────────────────
                    # The new leaf is placed as a sister to NN, with the same branch
                    # length as NN from the parent. This preserves the ultrametric
                    # property of UPGMA — all leaves in a clade remain equidistant
                    # from their common ancestor.
                    Parent_Node = self._find_parent(Placed_Tree, nn_name)
                    NN_Node     = next(Placed_Tree.find_clades(nn_name))

                    nn_branch_length = NN_Node.branch_length or 0.01
                    New_leaf = Clade(name=elem, branch_length=nn_branch_length)
                    Parent_Node.clades.append(New_leaf)

                elif placement_option == "B":
                    # ── Option B: split NN edge ───────────────────────────────
                    Parent_Node = self._find_parent(Placed_Tree, nn_name)
                    NN_Node     = next(Placed_Tree.find_clades(nn_name))

                    #forked statement just in case the node happens to be root (highly unlikely)
                    original_bl = NN_Node.branch_length if NN_Node.branch_length is not None else 0.01
                    split       = float(np.clip(split_pos, 0.0, original_bl))

                    # shorten existing NN branch
                    NN_Node.branch_length = original_bl - split

                    # insert intermediate node at split point
                    intermediate = Clade(
                        branch_length=split,
                        clades=[NN_Node, Clade(name=elem, branch_length=branch_length)])
                    # replace NN in parent's child list
                    idx = Parent_Node.clades.index(NN_Node)
                    Parent_Node.clades[idx] = intermediate

                elif placement_option == "C":
                    # ── Option C: attach from parent ─────────────────────────
                    Parent_Node = self._find_parent(Placed_Tree, nn_name)
                    New_leaf    = Clade(name=elem, branch_length=branch_length)
                    Parent_Node.clades.append(New_leaf)

                else:
                    raise ValueError(
                        f"Unknown placement_option '{placement_option}'. "
                        f"Use 'C' or 'B'.")

            except (ValueError, IndexError) as e:
                print(f"Warning: could not place '{elem}' — {e}. Skipping.")
                continue

        return Placed_Tree
