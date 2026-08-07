import numpy as np
import json
import pickle
import h5py
import os
import time
from pathlib import Path


# ===============================================================================
# UNIFIED INTERFACE (Has options to save as any of the preferred saving methods
# ===============================================================================
class DataPropertiesSaver:
    """
    Unified interface for saving/loading Data Properties
    
    Automatically chooses best format based on file extension.
    Defaults to Pickle (fastest and most reliable).
    
    USAGE:
    ------
    # Save
    saver = DataPropertiesSaver()
    saver.save(data_properties, 'saved_data/partition_0.pkl')
    
    # Load
    data_properties = saver.load('saved_data/partition_0.pkl')
    """
    
    def __init__(self, default_format='pickle'):
        """
        Args:
            default_format: 'pickle', 'hdf5', or 'npz'
        """
        self.default_format = default_format
    
    def save(self, data_properties, filepath):
        """
        Save Data Properties (auto-detects format from extension)
        
        Args:
            data_properties: Dictionary from mutant sorter
            filepath: Path with extension (.pkl, .h5, .npz)
        """
        # Detect format from extension
        ext = Path(filepath).suffix.lower()
        
        if ext == '.pkl' or ext == '.pickle':
            DataPropertiesSaver_Pickle.save(data_properties, filepath)
        elif ext == '.h5' or ext == '.hdf5':
            DataPropertiesSaver_HDF5.save(data_properties, filepath)
        elif ext == '.npz':
            DataPropertiesSaver_NPZ.save(data_properties, filepath)
        else:
            # Default to pickle
            print(f"Unknown extension {ext}, using pickle format")
            if not filepath.endswith('.pkl'):
                filepath += '.pkl'
            DataPropertiesSaver_Pickle.save(data_properties, filepath)
    
    def load(self, filepath):
        """
        Load Data Properties (auto-detects format from extension)
        
        Args:
            filepath: Path to saved file
            
        Returns:
            data_properties: Dictionary ready to use
        """
        # Detect format from extension
        ext = Path(filepath).suffix.lower()
        
        if ext == '.pkl' or ext == '.pickle':
            return DataPropertiesSaver_Pickle.load(filepath)
        elif ext == '.h5' or ext == '.hdf5':
            return DataPropertiesSaver_HDF5.load(filepath)
        elif ext == '.npz':
            return DataPropertiesSaver_NPZ.load(filepath)
        else:
            raise ValueError(f"Unknown file format: {ext}")
    
    def save_all_partitions(self, model, save_dir, weighted_out_mutants=False, format='pickle'):
        """
        Convenience method: Save data properties for ALL partitions
        
        Args:
            model: Your Compressed_Potts instance
            save_dir: Directory to save files
            weighted_out_mutants: Pass to mutant sorter
            format: 'pickle', 'hdf5', or 'npz'
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        extensions = {'pickle': '.pkl', '.hdf5': '.h5', 'npz': '.npz'}
        ext = extensions.get(format, '.pkl')
        
        print(f"Saving data properties for all {model.num_of_partitions} partitions...")
        start_time = time.time()
        
        for p in range(model.num_of_partitions):
            print(f"\nPartition {p}/{model.num_of_partitions - 1}...")
            
            # Generate data properties
            data_props = model.Fast_Single_Mutants_Sorter_Partition_for_Training_Data_OPTIMIZED(
                p, weighted_out_mutants)
            
            # Save
            filepath = os.path.join(save_dir, f'partition_{p}{ext}')
            self.save(data_props, filepath)
        
        elapsed = time.time() - start_time
        print(f"Saved all partitions in {elapsed:.2f} seconds")
        print(f"   Location: {save_dir}")
    
    def load_partition(self, save_dir, partition_index, format='pickle'):
        """
        Convenience method: Load data properties for specific partition
        
        Args:
            save_dir: Directory where files are saved
            partition_index: Which partition to load
            format: 'pickle', 'hdf5', or 'npz'
            
        Returns:
            data_properties: Ready to use in training
        """
        extensions = {'pickle': '.pkl', 'hdf5': '.h5', 'npz': '.npz'}
        ext = extensions.get(format, '.pkl')
        
        filepath = os.path.join(save_dir, f'partition_{partition_index}{ext}')
        return self.load(filepath)




###------------------- Saving method for each option:


class DataPropertiesSaver_Pickle:
    """
    Save and load Data Properties using Pickle
    
    RECOMMENDED: Fast, simple, handles your data structure perfectly.
    """
    
    @staticmethod
    def save(data_properties, filepath):
        """
        Save Data Properties to a pickle file
        
        Args:
            data_properties: Dictionary from Fast_Single_Mutants_Sorter_Partition_for_Training_Data_OPTIMIZED
            filepath: Path to save file (e.g., 'data_props_partition_0.pkl')
        """
        import pickle
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with highest protocol for speed/compression
        with open(filepath, 'wb') as f:
            pickle.dump(data_properties, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(filepath) / (1024**2)  # MB
        print(f"✅ Saved Data Properties to {filepath}")
        print(f"   File size: {file_size:.2f} MB")
    
    @staticmethod
    def load(filepath):
        """
        Load Data Properties from a pickle file
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            data_properties: Dictionary ready to use in training
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data_properties = pickle.load(f)
        
        print(f"✅ Loaded Data Properties from {filepath}")
        
        # Verify structure
        if 'Partition_index' in data_properties:
            print(f"   Partition: {data_properties['Partition_index']}")
        if 'Properties' in data_properties:
            n_sequences = len(data_properties['Properties'])
            print(f"   Sequences: {n_sequences}")
        
        return data_properties


# ============================================================
# METHOD 2: HDF5 (BEST FOR VERY LARGE DATA)
# ============================================================

class DataPropertiesSaver_HDF5:
    """
    Save and load Data Properties using HDF5
    
    Use this if your data is very large (>1GB) or you want language-agnostic format.
    Requires: pip install h5py --break-system-packages
    """
    
    @staticmethod
    def save(data_properties, filepath):
        """
        Save Data Properties to HDF5 file
        
        Args:
            data_properties: Dictionary from mutant sorter
            filepath: Path to save file (e.g., 'data_props_partition_0.h5')
        """
        import h5py
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save partition index
            f.attrs['Partition_index'] = data_properties['Partition_index']
            
            # Create group for properties
            props_group = f.create_group('Properties')
            
            # Iterate through each sequence
            for hash_val, props in data_properties['Properties'].items():
                # Create group for this sequence (use string key for HDF5)
                seq_group = props_group.create_group(str(hash_val))
                
                # Save each array
                for key, value in props.items():
                    if isinstance(value, np.ndarray):
                        seq_group.create_dataset(key, data=value, compression='gzip')
                    else:
                        # Store scalars as attributes
                        seq_group.attrs[key] = value
        
        file_size = os.path.getsize(filepath) / (1024**2)  # MB
        print(f"✅ Saved Data Properties to {filepath}")
        print(f"   File size: {file_size:.2f} MB (compressed)")
    
    @staticmethod
    def load(filepath):
        """
        Load Data Properties from HDF5 file
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            data_properties: Dictionary ready to use
        """
        import h5py
        
        data_properties = {}
        
        with h5py.File(filepath, 'r') as f:
            # Load partition index
            data_properties['Partition_index'] = f.attrs['Partition_index']
            
            # Load properties
            properties = {}
            props_group = f['Properties']
            
            for hash_str in props_group.keys():
                hash_val = int(hash_str)  # Convert back to int
                seq_group = props_group[hash_str]
                
                # Load arrays and attributes
                seq_props = {}
                
                # Load datasets (arrays)
                for key in seq_group.keys():
                    seq_props[key] = seq_group[key][:]  # Load array
                
                # Load attributes (scalars)
                for key in seq_group.attrs.keys():
                    seq_props[key] = seq_group.attrs[key]
                
                properties[hash_val] = seq_props
            
            data_properties['Properties'] = properties
        
        print(f"   Loaded Data Properties from {filepath}")
        print(f"   Partition: {data_properties['Partition_index']}")
        print(f"   Sequences: {len(data_properties['Properties'])}")
        
        return data_properties


# ============================================================
# METHOD 3: NUMPY NPZ (SIMPLE ALTERNATIVE)
# ============================================================

class DataPropertiesSaver_NPZ:
    """
    Save and load Data Properties using NumPy's .npz format
    
    Good middle ground - native NumPy, compressed, but needs some restructuring.
    """
    
    @staticmethod
    def save(data_properties, filepath):
        """
        Save Data Properties to .npz file
        
        Args:
            data_properties: Dictionary from mutant sorter
            filepath: Path to save file (e.g., 'data_props_partition_0.npz')
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten the nested structure for npz
        arrays_to_save = {}
        
        # Save partition index
        arrays_to_save['partition_index'] = np.array([data_properties['Partition_index']])
        
        # Save hash values
        hash_values = list(data_properties['Properties'].keys())
        arrays_to_save['hash_values'] = np.array(hash_values)
        
        # Save each property for each sequence
        for i, hash_val in enumerate(hash_values):
            props = data_properties['Properties'][hash_val]
            prefix = f'seq_{i}_'
            
            for key, value in props.items():
                if isinstance(value, np.ndarray):
                    arrays_to_save[prefix + key] = value
                else:
                    arrays_to_save[prefix + key] = np.array([value])
        
        # Save with compression
        np.savez_compressed(filepath, **arrays_to_save)
        
        file_size = os.path.getsize(filepath) / (1024**2)  # MB
        print(f"   Saved Data Properties to {filepath}")
        print(f"   File size: {file_size:.2f} MB")
    
    @staticmethod
    def load(filepath):
        """
        Load Data Properties from .npz file
        
        Args:
            filepath: Path to .npz file
            
        Returns:
            data_properties: Dictionary ready to use
        """
        # Load npz file
        data = np.load(filepath, allow_pickle=False)
        
        # Reconstruct data structure
        data_properties = {}
        data_properties['Partition_index'] = int(data['partition_index'][0])
        
        hash_values = data['hash_values']
        properties = {}
        
        for i, hash_val in enumerate(hash_values):
            prefix = f'seq_{i}_'
            seq_props = {}
            
            # Find all keys for this sequence
            for key in data.files:
                if key.startswith(prefix):
                    prop_name = key[len(prefix):]
                    value = data[key]
                    
                    # Unpack scalars
                    if value.shape == (1,) and prop_name in ['data_multiplicity']:
                        seq_props[prop_name] = int(value[0])
                    else:
                        seq_props[prop_name] = value
            
            properties[int(hash_val)] = seq_props
        
        data_properties['Properties'] = properties
        
        print(f"   Loaded Data Properties from {filepath}")
        print(f"   Partition: {data_properties['Partition_index']}")
        print(f"   Sequences: {len(data_properties['Properties'])}")
        
        return data_properties


