import numpy as np 
import math
from tqdm import tqdm


### Compressed Potts model with advanced training features
#-> Parallel Tempering
#-> Adaptive Learning
#-> persistent contrastive learning
#-> mixed and anchored chains
#-> various temperature schemes

class Compressed_LRBM:
    def __init__(self, Compact_Training_Data, Partition_Map = None, Influence = True, extra_character = "zz?",Temperature =1 ,noise = 1e-8, compressed= False):

        self.compact_data = Compact_Training_Data.copy()

        self.Total_data_size = len(Compact_Training_Data)
        self.num_of_positions = np.shape(Compact_Training_Data)[1]

        self.alphabet_list = np.unique(self.compact_data)
        self.total_alphabet_size = len(self.alphabet_list)
        self.compress = compressed #<=if false, we will not use compressed
        self.Influence = Influence
        self.Temperature = Temperature
        self.noise = noise
        
        ###
        if self.compress:
            # we need to introduce extra character
            self.extra_character = self.alphabet_list[-1] *2
            #self.extra_character = extra_character
            self.Check_type()
        self.compressed_alphabet_size , self.conversion_dictionary = self.Compressed_dictionary()
        self.num_of_positions = len(self.compressed_alphabet_size)

        ###-- constructing the list of indices for each position ---
        self.Position_index = self.Position_index_finder()

   
        ### construct fundamental connectivity and weights and biases
        #self.Init_Boltzmann_Parameters()

        ### --
        ## -- unfurling the data and building lookup tables ---
        self.unfurled_data = self.Unfurled_dataset(self.compact_data)
                ## finding the unique data points and data multiplicity 
        ##### assigning the unique data and multiplicity to respective class properties
        self.Unique_datapoints, self.Data_multiplicity, self.unique_data_args = self.Unique_data_finder(self.unfurled_data)

        ###------- Partition Setup ----- 
        if type(Partition_Map)==type(None):
            ### assume all-all connected
            self.Raw_Partition_Map = np.ones(self.num_of_positions).reshape(1,-1)
            self.num_of_partitions = 1
        else:
            self.Raw_Partition_Map = Partition_Map.copy()
            self.num_of_partitions = len(Partition_Map)
        print("Number of Partitions Set=", self.num_of_partitions)

        self.Partition_Map_Setter(self.Raw_Partition_Map)

            
        self.data_lookup, self.full_data_Hash_values = self.build_lookup_structure(self.unfurled_data)

        self.Init_Boltzmann_Parameters_ver2()

        ## the scorematching can get stuck to the mean solution very easily
        #self.Init_ScoreMatching_Scaled()


        ### Gauge fix
        for p in range(0, self.num_of_partitions):
            self.Bias[p] = self.Bias_Gauge_Fixing(p)

        ### 

    ### ----------- Helpers --------

    def build_lookup_structure(self, Data_Array):
        """Build optimized lookup structure once
        - For each UNIQUE data point as keys
        - the value will hold the probabilty (frequency/total_data_points)"""
        lookup = {}

        ## this contains hash values for all data points
        Hash_values = []

        length_of_data = len(Data_Array)
        for i, seq in enumerate(Data_Array):
            #hash(tuple(seq)) for memory efficiency
            # tuple(seq) is pretty inefficient
            key = hash(tuple(seq)) 
            Hash_values.append(key)
            if key in lookup:
                lookup[key] += 1
            else:
                lookup[key] = 1
        #return lookup, np.array(Hash_values)
        sorted_lookup_dict = dict(sorted(lookup.items()))
        Hash_values_sorted = np.array(list(sorted_lookup_dict.keys()))

        return sorted_lookup_dict, Hash_values_sorted
    

    def Hash_Value_given_data(self, Data_point):
        Value = hash(tuple(Data_point))
        return Value

    #def Hash_Values_batch(self, Data_points):
    #    """Batch hash computation for multiple data points"""
    #    # Method 1: Pure Python (good balance)
    #    return [hash(tuple(dp)) for dp in Data_points]
    
    def Hash_Values_batch(self, Data_points):
        """Fastest for pure Python"""
        return list(map(lambda dp: hash(tuple(dp)), Data_points))


    def Position_index_finder(self):
        """Constructs a dictionary of indices for each position in the one-hot encoded space"""
        Position_index = {}
        start_p = 0
        for p in range(0, len(self.compressed_alphabet_size)):
            end_p = start_p + self.compressed_alphabet_size[p]
            Position_index[p] = np.arange(start_p, end_p)
            start_p = end_p
        return Position_index
    

    def Unique_data_finder(self, Data):

        Data_Dict, datapoint_hash = self.build_lookup_structure(Data)
        #Unique_datapoint_hash = list(Data_Dict.keys())

        #Unique_datapoint_hash = np.array(list(Data_Dict.keys()))
        unique_datapoint_args = []
        for h in Data_Dict.keys():
            a_h = (np.argwhere(datapoint_hash==h)[0])
            unique_datapoint_args.append(a_h)
        unique_datapoint_args = np.array(unique_datapoint_args).reshape(-1)

        Multiplicity = np.array(list(Data_Dict.values()))
        Unique_datapoints  = Data[unique_datapoint_args]
        return Unique_datapoints, Multiplicity , unique_datapoint_args
        

    def Check_type(self):
        if self.extra_character not in self.alphabet_list:
            print("extra character=",self.extra_character )
            print("last alphabet =" , self.alphabet_list[-1])
            print("Extra character is okay.")

        else:
            ## last character is probably 0.
            m = np.random.randint(0, self.total_alphabet_size ,1)

            self.extra_character = self.alphabet_list[-1] +  self.alphabet_list[m]

            print("internal extra character was off")
            print("new extra character=",self.extra_character )
            print("last alphabet =" , self.alphabet_list[-1])
            self.Check_type()

    def Compressed_dictionary(self):
        ## will we compress it or not
        compression_status = self.compress

        dimension =[]
        conversion_dictionary ={}
        if compression_status:
            # we will compress it
            for i,c in enumerate(self.compact_data.T):
                unique_i = np.unique(c)
                dim_i = len(unique_i)+1
                unique_i = np.concatenate((unique_i, [self.extra_character]))
                conversion_dictionary[i] = unique_i
                dimension.append(dim_i)
        else:
            # we will not compress 
            Unique_set = self.alphabet_list
            for i in range(0, self.num_of_positions):
                unique_i = Unique_set
                dim_i = len(unique_i)
                conversion_dictionary[i] = Unique_set
                dimension.append(dim_i)

        return np.array(dimension), conversion_dictionary
    

    def Partition_Map_Setter(self, Raw_Partition_Map):
        print("..Setting Partition Map..")
        print("Number of partitions =", len(Raw_Partition_Map))

        """Sets the raw partition map
        - For each partition in the raw partitions, 
        constructs a list of indices pertaining to data in that partition in 1-hot encoded bases"""

        self.Raw_Partition_Map = Raw_Partition_Map

        Connected_indices ={}
        for i,part_i in enumerate(Raw_Partition_Map):
            Arg_p = np.argwhere(part_i==1).reshape(-1)
            conn_p =[]
            for a in Arg_p:
                conn_p.append(self.Position_index[a])

            conn_p = np.concatenate(conn_p)
            Connected_indices[i] = conn_p

        self.Partition_Map = Connected_indices


        self.Init_Boltzmann_Parameters_ver2()
        
        #self.Init_ScoreMatching_Scaled()

        self.Hash_Value_by_partition, self.Lookup_table_by_partition = self.Partitioned_Data_Lookup_table()
        self.Partition_index_slice = self.Partition_Index_slicer()
        


    def One_hot_encoder(self, Compact_state):
        """For any given state, it returns a one hot encoded state
        - unseen words at any position are mapped to unknown"""
        Dictionary = self.conversion_dictionary
        Dim = self.compressed_alphabet_size
        Full_Vector = []
        for i,word_i in enumerate(Compact_state):
            dim_i = Dim[i]
            Dict_i = Dictionary[i]
            Sub_vector = np.zeros(dim_i)

            try:
                arg_one = (np.argwhere(Dict_i==word_i).reshape(-1))
            except:
                ## word out of vocabulary. Map it to unknown
                arg_one = (np.argwhere(Dict_i==self.extra_character).reshape(-1))

            Sub_vector[arg_one] =1
            Full_Vector.append(Sub_vector)
        Full_Vector = np.concatenate(Full_Vector)
        return Full_Vector
    

    def One_hot_decoder(self, Unfurled_state):
        """for any one hot encoded state, it returns the compact state
         (list of words that make the sentence)"""
        Dictionary = self.conversion_dictionary
        Dim = self.compressed_alphabet_size
        Compact_state = []
        start_index = 0
        end_index = 0
        for i, dim_i in enumerate(Dim):
            end_index += dim_i
            Dict_i = Dictionary[i]
            sub_vector = Unfurled_state[start_index:end_index]
            arg_one= np.argwhere(sub_vector==1).reshape(-1)
            word_i = Dict_i[arg_one]
            start_index += dim_i
            Compact_state.append(word_i)

        return np.array(Compact_state).reshape(-1)

    def Unfurled_dataset(self, Data):
        """Takes any compact dataset and writes it in a one-hot encoded compressed basis
        - Compressed meaning all the words never seen at a position are mapped to a single position in one hot encoded vector for that word"""
        Full_set = []
        for d in Data:
            unf_d = self.One_hot_encoder(d)
            Full_set.append(unf_d)
        return np.array(Full_set)


    
    def Fundamental_Connectivity(self):

        Partition_Map_dict = self.Partition_Map
        Position_index = self.Position_index
        Raw_partition = self.Raw_Partition_Map
        Influence = self.Influence

        ### this doesn't have zeroing out of the diagonals

        Fundamental_connectivity_dict = {}
        partitions = list(Partition_Map_dict.keys())

        for p in partitions:
            #raw_part = Raw_partition[p]
            #positions_p = np.argwhere(raw_part==1).reshape(-1)
            Total_dof_p_row = len(Partition_Map_dict[p])
            Total_dof_p_col = Total_dof_p_row
            Total_dof_prev = 0

            if p > 0 and Influence:
                Total_dof_prev = len(Partition_Map_dict[p-1])
                Total_dof_p_col+=Total_dof_prev

            FC_p = np.ones((Total_dof_p_row, Total_dof_p_col))

            Fundamental_connectivity_dict[p] = FC_p

        ### Now we zero out the appropriate blocks that correspond to self interaction.
        for idx in partitions:
            Raw_part = Raw_partition
            full_part = Partition_Map_dict
            Position_index = Position_index
            Super_offset = 0
            if idx>0 and Influence:
                Super_offset = len((full_part[idx-1]).reshape(-1))
            arg_part = np.argwhere(Raw_part[idx]==1).reshape(-1)
            len_part = len(full_part[idx])
            offset = 0
            FC_i = Fundamental_connectivity_dict[idx]
            for p in arg_part:
                #print(p)
                args_p = np.arange(0,len(Position_index[p])) + offset 
                args_p_col = args_p + Super_offset

                #print(len(args_p))
                #print(np.shape(FC))
                v1 = np.zeros(len_part) 
                v1[args_p] = 1

                v2 = np.zeros(Super_offset+len_part)
                v2[args_p_col] = 1
                
                Matrix = np.outer(v1, v2)
                offset += len(Position_index[p])
                FC_i-= Matrix

        
        return Fundamental_connectivity_dict

    #----------------------------- Data Partition into self and interaction slices ------------------------------------

    
    def Data_part_by_index(self, partition_index):
        """Partitions the Unique training Data on the given partition
        Input:
        partition_index(int): from 0 to len(Partition)
        """
        partition_index = int(partition_index)
        Data = self.Unique_datapoints
        Part_args = self.Partition_Map[partition_index]
        Data_part = Data[:,Part_args]
        return Data_part
    
    def Data_part(self):
        """returns all partitions of the training data in dictionary form with partition index as keys"""
        Partitions = list(self.Partition_Map.keys())
        Partitioned_Data = {}
        for p in Partitions:
            Partitioned_Data[p] = self.Data_part_by_index(p)
        return Partitioned_Data
    
    def Data_Partition(self, Data, partition_index):
        #Data = self.Unique_datapoints
        padded = False
        if len(np.shape(Data)) ==1:
            ## we will need to pad the other dimension for slicing
            Data = Data.reshape(1,-1)
            padded = True

        Row_args = self.Partition_Map[partition_index]

        if self.Influence and partition_index>0:
            Col_args = np.concatenate([self.Partition_Map[partition_index-1],self.Partition_Map[partition_index]])

        else:
            Col_args = self.Partition_Map[partition_index]

        Data_self = Data[:, Row_args]
        Data_interaction = Data[:, Col_args]

        if padded==True:
            Data_self = Data_self.reshape(-1)
            Data_interaction = Data_interaction.reshape(-1)

        return Data_self, Data_interaction

    def Data_Partition_args_and_multiplicity(self, Data, partition_index):
        Data_Self, Data_Interaction = self.Data_Partition(Data, partition_index)
        Data_interact_unique, Multiplicity_unique, Arg_unique = self.Unique_data_finder(Data_Interaction)
        Data_Self_unique = Data_Self[Arg_unique.reshape(-1)]
        return  Data_Self_unique, Data_interact_unique, Multiplicity_unique, Arg_unique
    
    def Partitioned_Data_Lookup_table(self):

        """For the entire dataset, for each partition, this constructs a hash-value dictionary and a lookup table
        - Basically lookup table for each partition rather than the whole dataset"""

        Hash_values_by_partition={}
        Lookup_tables_by_partition = {}

        for p in range(0, self.num_of_partitions):
            Data_in_partition_self, Data_in_partition_with_influence = self.Data_Partition(self.unfurled_data, p)
            Lookup_table_p, Hash_values_p = self.build_lookup_structure(Data_in_partition_with_influence)
            Hash_values_by_partition[p] = Hash_values_p
            Lookup_tables_by_partition[p] = Lookup_table_p

        return Hash_values_by_partition, Lookup_tables_by_partition
    
    def Partition_Index_slicer(self):

        Raw_Partition = self.Raw_Partition_Map
        Compressed_alphabet_size = self.compressed_alphabet_size

        Index_slices={}
        for p in range(0,len(Raw_Partition)):
            partition = Raw_Partition[p]
            positions_p = np.argwhere(partition==1).reshape(-1)
            Alphabets_size_p = Compressed_alphabet_size[positions_p]
            cumulative_size = np.cumsum(Alphabets_size_p)
            start=0
            Slice_p={}
            for i in range(0, len(Alphabets_size_p)):
                end = int(cumulative_size[i])
                Slice_p[int(positions_p[i])] = [start, end]
                #Slice_p.append([start, end])
                start = end

            Index_slices[p] = Slice_p

        return Index_slices
    # ------------------------------------------------------------------------------------
    
    ##------------------------ Different types of initializations-------------------------

    def Init_Boltzmann_Parameters(self):
        """
        ========================= Cold start =======================
         -Initialize with symmetric W_self that has random values
         - Initialize with random bias vector
        ============================================================
        """

        FC = self.Fundamental_Connectivity()
        self.fundamental_connectivity = FC
        
        Partitions = list(FC.keys())
        Weights = {}
        Bias = {}
        
        Data_mean = np.mean(self.unfurled_data, axis=0)
        
        for p in Partitions:
            Row_size, Col_size = np.shape(FC[p])
            Weight_p = np.random.normal(loc=0, scale=100*self.noise, size=(Row_size, Col_size))
            Weight_p *= FC[p]
            
            # ============================================================
            # Make W_self symmetric
            # ============================================================
            split_idx = (Col_size - Row_size) if (self.Influence and p > 0) else 0
            W_self = Weight_p[:, split_idx:]
            Weight_p[:, split_idx:] = (W_self + W_self.T) / 2
            # ============================================================
            
            p_indices = self.Partition_Map[p]
            #Data_mean_p = Data_mean[p_indices] * (-0.01)
            frequencies = np.clip(Data_mean[p_indices], 1e-5, 1)
            Bias_p = self.Temperature * np.log(frequencies) * 1e-5
            Bias_p += np.random.normal(loc=0, scale=self.noise, size=Row_size)
            
            
            Weights[p] = Weight_p
            Bias[p] = Bias_p
        
        self.Weights = Weights
        self.Bias = Bias
        
        self.Data_Partitioned = self.Data_part()
        self.partition_multiplicity_arg_dict = self.Data_arg_multiplicity_dictionary_constructor(self.unfurled_data)
    
    def Init_Boltzmann_Parameters_ver2(self):
        """
        ====================== Semi warm start ===========================
         - Initialize with symmetric W_self that has random values
         - Initialize with bias value that matches the log(frequencies)
         The bias here is the profile model solution
        ==================================================================
        """
        print("semi-warm start. Matching only the frequencies")
        FC = self.Fundamental_Connectivity()
        self.fundamental_connectivity = FC
        
        Partitions = list(FC.keys())
        Weights = {}
        Bias = {}
        
        Data_mean = np.mean(self.unfurled_data, axis=0)
        
        for p in Partitions:
            Row_size, Col_size = np.shape(FC[p])
            p_indices = self.Partition_Map[p]
            
            # Initializing bias from frequencies
            frequencies = np.clip(Data_mean[p_indices], 1e-8, 1.0)
            Bias_p = self.Temperature * np.log(frequencies)
            
            # Measure typical bias magnitude
            bias_magnitude = np.mean(np.abs(Bias_p))
            
            # Initialize weights to be 20% of bias magnitude
            # This allows weights to have effect without dominating
            weight_std = 0.05 * bias_magnitude  # Adjust 0.2 to 0.1-0.5 as needed
            
            Weight_p = np.random.normal(loc=0, scale=weight_std, size=(Row_size, Col_size))
            Weight_p *= FC[p]
            
            # Make W_self symmetric
            split_idx = (Col_size - Row_size) if (self.Influence and p > 0) else 0
            W_self = Weight_p[:, split_idx:]
            Weight_p[:, split_idx:] = (W_self + W_self.T) / 2
            
            # Add tiny noise to bias for symmetry breaking
            Bias_p += np.random.normal(loc=0, scale=self.noise, size=Row_size)
            
            Weights[p] = Weight_p
            Bias[p] = Bias_p * 0.1
        
        self.Weights = Weights
        self.Bias = Bias



    def Init_ScoreMatching_Scaled(self):
        """
        ====================== Warm start  ======================
        Score matching / pseudolikelihood: 
        This is the analytical solution for score matching.
        - the weight matrix = pairwise correlation 
        - the weight vector = frequencies
        - small random noise added to both
        =========================================================
        """        
        print("Initializing with score matching ...")
        
        Data = self.unfurled_data
        N = len(Data)
        Data_mean = np.mean(Data, axis=0)
        
        # Compute correlations
        Correlation = (Data.T @ Data) / N
        Connected_Corr = Correlation - np.outer(Data_mean, Data_mean)
        
        
        FC = self.Fundamental_Connectivity()
        self.fundamental_connectivity = FC
        
        Weights = {}
        Bias = {}
        
        for p in range(self.num_of_partitions):
            p_indices = self.Partition_Map[p]
            
            if self.Influence and p > 0:
                prev_indices = self.Partition_Map[p-1]
                all_indices = np.concatenate([prev_indices, p_indices])
            else:
                all_indices = p_indices
            
            C_block = Connected_Corr[np.ix_(p_indices, all_indices)]
            
            # Weights from correlations (pseudolikelihood analytical solution)
            Weight_p = -C_block / (1 + 0.1)
            Weight_p *= FC[p]
            
            # Bias from frequencies
            frequencies = np.clip(Data_mean[p_indices], 1e-8, 1.0)
            #Bias_p = self.Temperature * np.log(frequencies)
            Bias_p = -1*frequencies 
        
            
            # Make W_self symmetric
            Row_size, Col_size = Weight_p.shape
            split_idx = (Col_size - Row_size) if (self.Influence and p > 0) else 0
            W_self = Weight_p[:, split_idx:]
            Weight_p[:, split_idx:] = (W_self + W_self.T) / 2
            
            Bias_p += np.random.normal(loc=0, scale=self.noise, size=len(p_indices))
            
            Weights[p] = Weight_p
            Bias[p] = Bias_p
        self.Weights = Weights
        self.Bias = Bias

    #############
    def Bias_Gauge_Fixing(self, partition_index):
        Bias = self.Bias[partition_index].copy()
        partition_indices = np.argwhere(self.Raw_Partition_Map[partition_index]==1).reshape(-1)
        Alphabet_sizes= self.compressed_alphabet_size
        current_idx = 0

        Bias_new = np.zeros_like(Bias)
        for i in partition_indices:
            start_index = current_idx
            end_index = current_idx + Alphabet_sizes[i]
            Bias_i = Bias[start_index:end_index]
            Mean_i = np.mean(Bias_i)
            
            Bias_new[start_index:end_index] = Bias_i - Mean_i
            ## shift current to next position
            current_idx+=Alphabet_sizes[i]
        return Bias_new


    ### ============================ Mutant generator functions  ===================================
   

    ### ================================== Energy Calculators ======================================
    def Energy_Array_given_exact_slices(self, Weight_part, Bias_part, Data_self, Data_interaction):
        """ Input : Weight matrix for a partition
                  : Bias vector for a partition
                  : Data self, Data interaction = self energy as well as interaction (previous partition+ current partition of data)

            Output: energy for an Array using partitioned weight/bias matrices and slices of the data

         -> This is inconvenient to use as a stand alone. 
         -> But it save some time when running the calculation for the same data slices.
         -> You don't have to keep fragmenting the data or searching for weight and bias pieces over and over"""

        Energy_all = np.sum(Data_self @ Weight_part * Data_interaction , axis = 1) + Data_self@Bias_part
        return Energy_all


    def Energy_Array_given_partition_easy(self, Unfurled_Array, partition_index):

        """ Input : Unfurled Data array
                  : Partition index for which energy is being calculated

            Output:  Energy for an Array in any partition"""

        Data_self, Data_interaction = self.Data_Partition(Unfurled_Array, partition_index)
        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]
        Energy_all = np.sum(Data_self @ Weight_part * Data_interaction , axis = 1) + Data_self@Bias_part
        return Energy_all
    

    def Energy_state_given_partition(self, Unfurled_Sequence, partition_index):
        
        """ Input : Unfurled sequence
                  : Partition index for which energy is being calculated

            Output:  Energy for the state in any partition"""

        Data_self, Data_interaction = self.Data_Partition(Unfurled_Sequence, partition_index)

        Weight = self.Weights[partition_index]
        Bias = self.Bias[partition_index]
        
        Energy_Sequence = np.dot(Data_self @ Weight, Data_interaction) + np.dot(Bias, Data_self)
        
        return Energy_Sequence


    
    def Energy_Mutant_Array_light(self, Weight_part, Bias_part, Mutant_self_full, D_prev):
        """Calculates energy for a Mutant Array using partitioned weight/bias matrices
        - This needs to have mutant arrays otherwise it doesn't work
        - Speed comes from the fact that the influencing piece for all Mutant is the same
        - Hence calculation can be fragmented into self energy + influence energy"""
        a, b = np.shape(Weight_part)
        W_self = Weight_part[:, (b-a):]
        W_interact = Weight_part[:, 0:(b-a)]

        Bias_Influence = (W_interact @ D_prev) + Bias_part

        E_total = np.sum(Mutant_self_full @ W_self * Mutant_self_full, axis=1) + Mutant_self_full @ Bias_Influence
        return E_total
    
    #####----------------------------------------------------------------------------------------

    def Data_arg_multiplicity_dictionary_constructor(self, Data):
        
        Full_arg_multiplicity_dict={}

        for p in range(0, len(self.Raw_Partition_Map)):
            p_dict ={}
            Data_self_unique,Data_interact_unique, mult_p, arg_p = self.Data_Partition_args_and_multiplicity(Data,p)
            p_dict['Number_of_unique_data'] = len(arg_p)
            p_dict["Data_self_unique"] = Data_self_unique
            p_dict["Data_interact_unique"] = Data_interact_unique
            p_dict["Multiplicity"] = mult_p
            p_dict["Args"] = arg_p

            
            Full_arg_multiplicity_dict[p] = p_dict
            
        return Full_arg_multiplicity_dict
            
    ###########
    def Random_state_generator(self, verbose = False):

        """Generated a random one-hot encoded state
         - If verbose = true, it returns - one-hot-encoded state, compact state
         - Else , it returns one-hot encoded state """

        words = np.random.choice(self.alphabet_list, self.num_of_positions)

        one_hot_random = self.One_hot_encoder(words)
        if verbose:
            return one_hot_random, words
        else:
            return one_hot_random

    ### -----------------
        ### -----------------For activation Probabilities --------------------
    
    def Softmax_probability_stable(self, Energy_list, sign = -1, temperature=1):
    
        """================ stable softmax function that doesn't overflow =================
        -> if sign = 1 --> softmax of Energy_list 
        -> if sign = -1 --> softmin of Energy_list (Default)
        -> temperature scales energy by division (default temperature = 1)
        !!! sign set to softmin as default according to Energy convention in stat mech
                --> Low energy = high probability (Softmin function)
        """

        Energy_array = np.array(Energy_list) * sign

        max_value = np.max(Energy_array)

        Energy_array_scaled = (Energy_array - max_value)/temperature

        likelihood_energy = np.exp(Energy_array_scaled)

        normalization = np.sum(likelihood_energy)

        Softmax = likelihood_energy/normalization

        return Softmax


    def Single_Mutants_Generator_Subset(self, Unfurled_Sequence, partition_index, positions):
        Data_self, Data_interaction = self.Data_Partition(Unfurled_Sequence, partition_index)
        Diff = len(Data_interaction) - len(Data_self)
        Connectivity_partition = self.fundamental_connectivity[partition_index]
        Part_dof = len(Data_self)
        Data_prev_part = Data_interaction[:Diff]

        Index_slice_dict = self.Partition_index_slice[partition_index]
        dof_indices = np.concatenate([np.arange(*Index_slice_dict[p]) for p in positions])

        # only pull the rows we need instead of building the full Part_dof x Part_dof matrix
        Connectivity_relevant = Connectivity_partition[dof_indices, Diff:]

        num_mutants = len(dof_indices)
        eye_subset = np.zeros((num_mutants, Part_dof))
        eye_subset[np.arange(num_mutants), dof_indices] = 1

        Single_mutant_self = (Connectivity_relevant * Data_self) + eye_subset

        return Single_mutant_self, Data_prev_part

    def Activation_Probability_Subset(self, Unfurled_Sequence, partition_index, positions, temperature=None):
        if temperature == None:
            temperature = self.Temperature
        else:
            temperature = temperature

        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]

        Data_self, D_prev = self.Single_Mutants_Generator_Subset(Unfurled_Sequence, partition_index, positions)
        E_mutants = self.Energy_Mutant_Array_light(Weight_part, Bias_part, Data_self, D_prev)

        Probability_array = []
        start = 0
        Positions_end = np.cumsum(self.compressed_alphabet_size[positions])
        for p_end in Positions_end:
            E_site = E_mutants[start:p_end]
            Prob_site = self.Softmax_probability_stable(E_site, sign=-1, temperature=temperature)
            start = p_end
            Probability_array.append(Prob_site)

        return np.concatenate(Probability_array)


    def Forward_Gibbs_Pass(self, Unfurled_Sequence, partition_index, Step_Split=0.1, temperature=1):
        Positions_in_partition = list(self.Partition_index_slice[partition_index].keys())
        Index_slice_dict = self.Partition_index_slice[partition_index]

        Length = len(Positions_in_partition)
        Num_sites_to_change = max(1, int(Length * Step_Split))
        Positions_to_change = np.sort(np.random.choice(Positions_in_partition, Num_sites_to_change, replace=False))

        Activation_prob = self.Activation_Probability_Subset(Unfurled_Sequence, partition_index, Positions_to_change, temperature=temperature)

        start = 0
        for pos in Positions_to_change:
            indices_pos = self.Position_index[pos]
            pos_start, pos_stop = Index_slice_dict[pos]
            num_alphabets = pos_stop - pos_start

            Prob_i = Activation_prob[start:start + num_alphabets]
            start += num_alphabets

            Picked_state = np.random.choice(range(num_alphabets), p=Prob_i)
            Next_state = np.zeros(len(indices_pos))
            Next_state[Picked_state] = 1
            Unfurled_Sequence[indices_pos] = Next_state

        return Unfurled_Sequence

    def Gibbs_Sampling_Partitioned(self, State, partition_index, Num_iterations, Step_Split = 0.1, temperature = 1):
        if temperature == None:
            temperature = self.Temperature
        else:
            temperature = temperature
    
        State = State.copy()
        for i in range(0, Num_iterations):
            State = self.Forward_Gibbs_Pass(State,partition_index,  Step_Split, temperature=temperature)
        return State
    

    def Gibbs_Sampling_through_Layers(self, Num_iterations, Step_Split=0.1, temperature = 1):
        if temperature == None:
            temperature = self.Temperature
        else:
            temperature = temperature
        Random_State = self.Random_state_generator()
        Final_State = Random_State.copy()

        for partition_index in range(0,len(self.Raw_Partition_Map)):
            #Row_args, Col_args = self.Relevant_Partition_Args(p_index)
            #W_part, B_part = self.Weight_Bias_Partition(Row_args, Col_args)
            Final_State = self.Gibbs_Sampling_Partitioned(Final_State, partition_index, Num_iterations, Step_Split, temperature=temperature)
        return Final_State, Random_State


    ###------------------- Implementing Parallel tempering ---------------

    ### We will need a lot of duplicate looking functions
    ### but they are here for specific application inside the parallel tempering routine for faster runtile.
    def Random_Data_Self_Local(self, partition_index):
        """Generates a random one-hot-encoded Data_self for this partition"""
        Index_slice_dict = self.Partition_index_slice[partition_index]
        Positions_in_partition = list(Index_slice_dict.keys())

        Part_dof = sum(self.compressed_alphabet_size[p] for p in Positions_in_partition)
        Data_self_random = np.zeros(Part_dof)

        for pos in Positions_in_partition:
            pos_start, pos_stop = Index_slice_dict[pos]
            num_alphabets = pos_stop - pos_start
            choice = np.random.randint(0, num_alphabets)
            Data_self_random[pos_start + choice] = 1

        return Data_self_random

    def PT_Anchored_Slot_Indices(self, temperature_list):
        """
        chains   => (cold chain elementat sample.... slightly noisy chain element...^^...purely random chain element)
        This sets how many slightly noisy chain elements to keep. 
        ^^ = slot upto where slightly noisy chains are kept.

        Input:
        Temperature list for chains

        Output:
        Returns the chain slot indices (coldest half of the ladder) that
        Parallel_Tempering_Partition's graded anchoring seeds toward data on a bootstrap call.
        Centralized here so other code (e.g. an unbiased reconstruction-error diagnostic that
        wants to EXCLUDE anchored slots) can't drift out of sync with the actual anchoring logic.

        Given:
        -  temperature_list = the same ladder passed to Parallel_Tempering_Partition
        - Returns
            - anchored_indices = array of chain slot indices, coldest -> less-cold, that
                    get seeded at (increasingly noisy copies of) Anchor_State on a bootstrap call
        """
        num_chains = len(temperature_list)
        temps_arr = np.array(temperature_list, dtype=float)
        order = np.argsort(temps_arr)  # coldest -> hottest, by chain slot index
        num_anchored = max(1, num_chains // 2)
        return order[:num_anchored]
    

    def Perturb_Data_Self_Local(self, Data_self_state, partition_index, flip_fraction):
        """Randomly reassigns flip_fraction of positions in a partition-local Data_self
        to a UNIFORMLY random alternative letter (NOT weighted by the model) -- used to
        inject graded noise around an anchor state (e.g. the true data point) rather than
        either an exact copy of it or a fully independent random draw.
        Given:
        -  Data_self_state = the state to perturb (e.g. Anchor_State); NOT modified in place
        -  flip_fraction = fraction of positions to reassign; 0 returns an exact copy
        """
        Data_self_state = np.array(Data_self_state).copy()
        Index_slice_dict = self.Partition_index_slice[partition_index]
        Positions_in_partition = list(Index_slice_dict.keys())

        Length = len(Positions_in_partition)
        Num_sites_to_flip = int(round(Length * flip_fraction))
        if Num_sites_to_flip <= 0:
            return Data_self_state

        Positions_to_flip = np.random.choice(Positions_in_partition, Num_sites_to_flip, replace=False)

        for pos in Positions_to_flip:
            pos_start, pos_stop = Index_slice_dict[pos]
            num_alphabets = pos_stop - pos_start
            choice = np.random.randint(0, num_alphabets)
            Data_self_state[pos_start:pos_stop] = 0
            Data_self_state[pos_start + choice] = 1

        return Data_self_state

    def Single_Mutants_Generator_Local(self, Data_self_state, D_prev, partition_index, positions):
        """Builds single-mutant rows for `positions` from a partition-local state,
        given a fixed D_prev (no re-partitioning of a global sequence needed)."""
        Connectivity_partition = self.fundamental_connectivity[partition_index]
        Diff = len(D_prev)
        Part_dof = len(Data_self_state)
        Index_slice_dict = self.Partition_index_slice[partition_index]

        dof_indices = np.concatenate([np.arange(*Index_slice_dict[p]) for p in positions])
        Connectivity_relevant = Connectivity_partition[dof_indices, Diff:]

        num_mutants = len(dof_indices)
        eye_subset = np.zeros((num_mutants, Part_dof))
        eye_subset[np.arange(num_mutants), dof_indices] = 1

        Single_mutant_self = (Connectivity_relevant * Data_self_state) + eye_subset
        return Single_mutant_self


    def Activation_Probability_Local(self, Data_self_state, D_prev, partition_index, positions, temperature=None):
        temperature = self.Temperature if temperature is None else temperature
        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]

        Mutant_self_full = self.Single_Mutants_Generator_Local(Data_self_state, D_prev, partition_index, positions)
        E_mutants = self.Energy_Mutant_Array_light(Weight_part, Bias_part, Mutant_self_full, D_prev)

        Probability_array = []
        start = 0
        Positions_end = np.cumsum(self.compressed_alphabet_size[positions])
        for p_end in Positions_end:
            E_site = E_mutants[start:p_end]
            Prob_site = self.Softmax_probability_stable(E_site, sign=-1, temperature=temperature)
            start = p_end
            Probability_array.append(Prob_site)

        return np.concatenate(Probability_array)

    def Pseudo_LogLikelihood_Local(self, Data_self_state, D_prev, partition_index, temperature=1):
        """============== Pseudo-likelihood: =============== 
        for EVERY position in the partition, the model's own
        conditional probability of the TRUE letter at that position, given everything else in
        the partition fixed (via Activation_Probability_Local, which already computes exactly
        this: the softmax over each position's alphabet using the current weights).
        this is not affected by Parallel Tempering or no Gibbs sampling.
        This is purely a function of the current weights and the true data, so it isn't biased by how negative-phase chains
        were initialized. 
        -> We can think of this as the the envelope to true likelihood. (kinda! ~ both move together)
        Given:
        -  Data_self_state = the TRUE partition-local state (e.g. Data_self_true)
        -  D_prev = that example's fixed background influence
        -  partition_index = the partition being scored
        -  temperature = read at this temperature (default 1, the model's own)

        - Returns
            - mean_neg_log_prob = average, over positions in the partition, of
                    -log( P(true letter at that position | rest of partition) )
                    --> LOWER is better (0 = model assigns full probability everywhere it should)
        """
        Index_slice_dict = self.Partition_index_slice[partition_index]
        Positions_in_partition = list(Index_slice_dict.keys())

        Activation_prob = self.Activation_Probability_Local(
            Data_self_state, D_prev, partition_index, Positions_in_partition, temperature=temperature)

        neg_log_probs = []
        for pos in Positions_in_partition:
            pos_start, pos_stop = Index_slice_dict[pos]
            true_local_idx = np.argmax(Data_self_state[pos_start:pos_stop])
            p_true = Activation_prob[pos_start + true_local_idx]
            neg_log_probs.append(-np.log(np.clip(p_true, 1e-12, None)))

        return float(np.mean(neg_log_probs))


    def Forward_Gibbs_Pass_Local(self, Data_self_state, D_prev, partition_index, Step_Split=0.2, temperature=1):
        """Same as Forward_Gibbs_Pass, but works entirely in partition-local
        coordinates, with D_prev fixed rather than re-derived from a global state."""
        Data_self_state = Data_self_state.copy()
        Index_slice_dict = self.Partition_index_slice[partition_index]
        Positions_in_partition = list(Index_slice_dict.keys())

        Length = len(Positions_in_partition)
        Num_sites_to_change = max(1, int(Length * Step_Split))
        Positions_to_change = np.sort(
            np.random.choice(Positions_in_partition, Num_sites_to_change, replace=False))

        Activation_prob = self.Activation_Probability_Local(
            Data_self_state, D_prev, partition_index, Positions_to_change, temperature=temperature)

        start = 0
        for pos in Positions_to_change:
            pos_start, pos_stop = Index_slice_dict[pos]
            num_alphabets = pos_stop - pos_start

            Prob_i = Activation_prob[start:start + num_alphabets]
            start += num_alphabets

            Picked_state = np.random.choice(range(num_alphabets), p=Prob_i)
            Data_self_state[pos_start:pos_stop] = 0
            Data_self_state[pos_start + Picked_state] = 1

        return Data_self_state


    def Reset_PT_Chain_Cache(self, partition_index=None):
        """Clears the persistent chain cache used by Contrastive_Divergence_Partition.
        Given:
        -  partition_index = if given, only clears that partition's cached chains
                    --> if None (default), clears ALL partitions' caches

        Use this when the cached chains might no longer be a good starting point --
        e.g. after a large or unusual weight update, after re-initializing self.Weights,
        or if you just want CD's negative phase to start fresh (random, data-anchored)
        again rather than persisting.
        """
        if not hasattr(self, 'PT_chain_cache'):
            self.PT_chain_cache = {}
            return

        if partition_index is None:
            self.PT_chain_cache = {}
        else:
            self.PT_chain_cache[partition_index] = {}

    def Update_Best_Checkpoint(self, partition_index, PLL_error):
        """Compares PLL_error against the best seen so far for this partition (TRAINING-SET
        PLL, since that's what Contrastive_Divergence_Partition computes) and, if it's an
        improvement, snapshots self.Weights[partition_index] / self.Bias[partition_index].
        Given:
        -  partition_index = the partition just trained
        -  PLL_error = this epoch's PLL_error for that partition (lower = better)

        Note: this checkpoints on TRAINING PLL, not held-out/validation PLL. That's a
        reasonable proxy given this training loop's noise (subsampled negative phase, PT
        sampling variance, adaptive LR), but it isn't a guard against overfitting -- if you
        have held-out data, call self.Pseudo_LogLikelihood_Local on it directly instead and
        checkpoint on THAT.
        """
        if not hasattr(self, 'Best_PLL_error'):
            self.Best_PLL_error = {}
            self.Best_Weights = {}
            self.Best_Bias = {}

        current_best = self.Best_PLL_error.get(partition_index, np.inf)

        if PLL_error < current_best:
            self.Best_PLL_error[partition_index] = PLL_error
            self.Best_Weights[partition_index] = self.Weights[partition_index].copy()
            self.Best_Bias[partition_index] = self.Bias[partition_index].copy()
            return True

        return False

    def Restore_Best_Weights(self, partition_index=None):
        """Loads the best-checkpointed self.Weights / self.Bias (by lowest TRAINING PLL_error
        seen so far, see Update_Best_Checkpoint) back into the live model.
        Given:
        -  partition_index = if given, restores only that partition
                    --> if None (default), restores ALL partitions that have a checkpoint

        Does NOT restore self.squared_grad_weights/bias (the RMSProp accumulator state) --
        only the parameters themselves. If you keep training after restoring, the adaptive
        learning rate starts accumulating fresh from this point, which is usually preferable
        to carrying forward gradient-scale statistics from the trajectory you just abandoned.
        """
        if not hasattr(self, 'Best_Weights') or len(self.Best_Weights) == 0:
            print("No checkpointed weights found -- nothing to restore.")
            return

        targets = [partition_index] if partition_index is not None else list(self.Best_Weights.keys())

        for p in targets:
            if p not in self.Best_Weights:
                print(f"No checkpoint found for partition {p} -- skipped.")
                continue
            self.Weights[p] = self.Best_Weights[p].copy()
            self.Bias[p] = self.Best_Bias[p].copy()
            print(f"Restored partition {p} to PLL_error={self.Best_PLL_error[p]:.5g}")


    def PT_Temperature_Steps(self, num_chains , min_temp = 0.5, max_temp = 5, type = 1):

        """
        - Given
        num_chains = number of chains to create during parallel tempering
        min_temp = smallest temperature for the chain
        max_temp = largest temperature for the chain
        
        type: 1,2,3,4
        1 --> Inverse-linear steps in temperature between min_temp and max_temp
                (linear steps in inverse temperature)
                - This performs the best in my experience
                
        2--> Geometric steps between min_temp and max_temp
                - This is also good

        3--> Linear steps between min_temp and max_temp
                - this is pretty poor. Mostly high energy phases explored

        4--> Same temperaure for all chains.
                - All chains set at T=1
                - This is to explore the space using walkers of same energy (step size)
        """

        ## sort and regularize the temperatures
        min_temp, max_temp = np.sort([min_temp, max_temp])  +1e-6
        if type==1 or type > 4:
            ### Inverse-linear temperature scheme (best!)
            ### linear steps in beta (the inverse temperature)
            beta_low_temp = 1/min_temp
            beta_high_temp = 1/max_temp
            beta_steps = np.linspace(beta_low_temp, beta_high_temp, num_chains)
            Temp_steps = 1/beta_steps
        elif type ==2:
            ### Geometric temperature steps scheme (good)
            ratio = (max_temp/min_temp)**(1/(num_chains-1))
            r_step = np.arange(0, num_chains)
            Temp_steps = min_temp *(ratio **r_step)
        elif type ==3:
            ### linear temperature steps scheme (poor)
            Temp_steps =  np.linspace(min_temp, max_temp, num_chains)
        elif type ==4:
            #all temperatures same. Basically if we wanted surity in sampling 
            ## here all we are doing is exploring the space with walkers of same energy
            Temp_steps = np.ones(shape = num_chains)
        return Temp_steps



    def Parallel_Tempering_Partition(self, partition_index, temperature_list, num_steps=10, Step_Split=0.1,
                                      Initial_State=None, D_prev=None, Initial_Chains=None, Anchor_State=None,
                                      Anchor_Max_Noise=0.3):

        """Parallel tempering restricted to a single partition:
        Input:
        -  partition_index = the partition being sampled
        -  temperature_list --> (the number of chains = length of this list)
        -  num_steps = number of Gibbs sweeps to take per chain (default = 10)
        -  Step_Split = Max proportion of sites to flip per sweep (passed through to Forward_Gibbs_Pass_Local)
        -  Initial_State = full unfurled state to derive D_prev from (only used if D_prev is not given)
        -  D_prev = background influence from outside the partition, held FIXED for the whole run
                    --> pass this directly (e.g. already computed via a batched Data_Partition call)
                        to skip re-partitioning Initial_State
                    --> either Initial_State or D_prev must be given
        -  Initial_Chains = optional list/array of chain states to START from (one per temperature),
                    e.g. the chains returned by a PREVIOUS call for this same example
                    --> lets chains persist across epochs (persistent CD) instead of always
                        re-mixing from scratch; if given, Anchor_State is ignored, since the
                        chains already have somewhere better to start from than either
                        random noise or the data point
                    --> if None, all chains start at an independent random Data_self
                        (self.Random_Data_Self_Local), UNLESS Anchor_State is also given
        -  Anchor_State = optional Data_self-shaped state (e.g. the true data point) used to seed
                    the COLDEST HALF of the temperature ladder -- GRADED, not uniform:
                    --> the single coldest chain is anchored EXACTLY at Anchor_State (zero noise)
                    --> chains between coldest and the ladder's midpoint get Anchor_State with
                        INCREASING noise (see Perturb_Data_Self_Local), scaling linearly from 0
                        up to Anchor_Max_Noise at the midpoint
                    --> chains at/above the midpoint (the hotter half) are left fully
                        INDEPENDENT RANDOM, untouched by Anchor_State
                    --> only used when Initial_Chains is None
                    --> rationale: cold chains have the sharpest distribution and benefit most
                        from a warm start (hardest to mix from noise); hot chains mix fast
                        regardless of start, so keeping them independent preserves PT's actual
                        job -- discovering modes away from the data, which the cold chain then
                        borrows via swaps. Anchoring ALL chains near data would remove that
                        exploration and risks reinforcing CD's classic blind spot for spurious
                        (data-far) modes.
        -  Anchor_Max_Noise = flip-fraction used at the edge of the anchored (coldest) half,
                    passed to Perturb_Data_Self_Local (default = 0.3)

        - Each chain, if not seeded above, starts at an INDEPENDENT random configuration of
          Data_self, all chains share the same fixed D_prev
        - Swaps between adjacent-temperature chains are attempted once per sweep,
          using the exponential of the energy/temperature difference (hardcoded to avoid overflow)

        - Output
            - chains = array of chain states (num_chains number of states), one per temperature
            - D_prev = the fixed background vector used for this run
                    --> returned so callers (e.g. contrastive divergence) can reuse
                        the exact same D_prev the chains were sampled under
        """
        if D_prev is None:
            if Initial_State is None:
                raise ValueError("Must provide either Initial_State or D_prev.")
            _, Data_interaction = self.Data_Partition(Initial_State, partition_index)
            Part_dof = len(self.Partition_index_slice[partition_index])  # or however Part_dof is derived
            Diff = len(Data_interaction) - Part_dof
            D_prev = Data_interaction[:Diff]

        num_chains = len(temperature_list)
        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]

        if Initial_Chains is not None:
            chains = [np.array(c).copy() for c in Initial_Chains]
        else:
            chains = [self.Random_Data_Self_Local(partition_index) for _ in range(num_chains)]
            if Anchor_State is not None:
                ## graded anchoring: coldest half of the ladder gets Anchor_State + increasing
                ## noise (coldest chain = exact copy, zero noise); hotter half stays fully random
                anchored_chain_indices = self.PT_Anchored_Slot_Indices(temperature_list)
                num_anchored = len(anchored_chain_indices)

                for rank, chain_idx in enumerate(anchored_chain_indices):
                    if num_anchored > 1:
                        noise_fraction = Anchor_Max_Noise * (rank / (num_anchored - 1))
                    else:
                        noise_fraction = 0.0

                    if noise_fraction <= 0:
                        chains[chain_idx] = np.array(Anchor_State).copy()
                    else:
                        chains[chain_idx] = self.Perturb_Data_Self_Local(Anchor_State, partition_index, noise_fraction)

        for step in range(num_steps):
            for i in range(num_chains):
                temp_for_chain = float(temperature_list[i])
                chains[i] = self.Forward_Gibbs_Pass_Local(
                    chains[i], D_prev, partition_index, Step_Split=Step_Split, temperature=temp_for_chain
                )

            energies = np.array([
                self.Energy_Mutant_Array_light(Weight_part, Bias_part, chains[i].reshape(1, -1), D_prev)[0]
                for i in range(num_chains)
            ])

            for i in range(num_chains - 1):
                j = i + 1
                Power = (energies[i] - energies[j]) / ((1 / temperature_list[i]) - (1 / temperature_list[j]))
                swap_factor = 1.0 if Power > 0 else np.exp(Power)
                if np.random.rand() < swap_factor:
                    chains[i], chains[j] = chains[j], chains[i]
                    energies[i], energies[j] = energies[j], energies[i]

        return np.array(chains), D_prev

    def Contrastive_Divergence_Partition(
        self,
        partition_index,
        State_Array,
        Temperature_list,
        num_chains,
        num_steps=10,
        Step_Split=0.5,
        batch_fraction=1.0,
        Bootstrap_num_steps=None,
        Anchor_Noise=0.3):

        """Contrastive divergence update for a single partition, using Parallel Tempering
        to draw negative-phase samples:
        Input:
        -  partition_index = the partition being trained
        -  State_Array = full unfurled data (any batch, duplicates allowed) used for BOTH phases
                    --> reduced internally to unique LOCAL examples for this partition via
                        self.Data_Partition_args_and_multiplicity, since two full sequences
                        that agree on this partition's D_prev and Data_self are identical
                        as far as this partition's energy function is concerned, even if
                        they differ elsewhere
        -  Temperature_list = temperature ladder passed to Parallel_Tempering_Partition
                               (the number of chains = length of this list)
        -  num_chains = number of PT chains per example
        -  num_steps = number of Gibbs sweeps per PT run, used whenever an example ALREADY
                    has cached chains to resume from (default = 10)
        -  Step_Split = Max proportion of sites to flip per sweep (default = 0.5)
        -  batch_fraction = fraction of the UNIQUE local examples used for the (expensive)
                             anti-Hebbian phase, RESAMPLED every call
                    --> the Hebbian phase is cheap (pure data statistics), so it always uses
                        all unique local examples, weighted by their local multiplicity P_eg
                    --> the anti-Hebbian phase requires running PT per example, so it's
                        restricted to a random subsample of unique examples, picked UNIFORMLY
                        for coverage; each picked example's gradient contribution is then
                        importance-weighted by its (renormalized) P_eg, consistent with the
                        Hebbian phase --> a common local context still counts for more than
                        a rare one, even though rare ones get an equal CHANCE to be sampled
        -  Bootstrap_num_steps = number of Gibbs sweeps used ONLY the first time an example is
                    seen (no cache entry yet, so chains start anchored rather than persisted)
                    --> defaults to 3 * num_steps if not given: a fresh anchor-seeded chain
                        needs more mixing to shed its initial bias than a chain that's already
                        near equilibrium from a previous epoch, so it's worth spending more
                        here even though it only happens once per example
        -  Anchor_Noise = Anchor_Max_Noise passed to Parallel_Tempering_Partition's graded
                    anchoring (coldest chain exact, coldest HALF of the ladder increasingly
                    noisy up to this fraction, hotter half fully random); only matters on
                    bootstrap calls, see Parallel_Tempering_Partition's docstring (default = 0.3)

        - Positive (Hebbian) phase: data-only statistics, independent of the current weights
                    --> computed once per call from the unique local examples

        - Negative (anti-Hebbian) phase: one Parallel_Tempering_Partition run per sampled example,
          under that example's own D_prev
                    --> corrections accumulated across the subsample, weighted by renormalized P_eg
                        (NOT a plain 1/Num_pick average)
                    --> PERSISTENT chains: each example's chains are cached in
                        self.PT_chain_cache[partition_index], keyed by a hash of [D_prev | Data_self],
                        and reused as the NEXT call's starting point instead of re-mixing from
                        random every epoch -- since weights move slowly, chains stay close to
                        equilibrium and only need light refreshing each epoch (persistent CD)
                    --> GRADED DATA-ANCHORED bootstrap: the FIRST time an example is seen (no
                        cache entry yet), the coldest half of the temperature ladder is seeded
                        at (increasingly noisy copies of) the true data point, the hotter half
                        stays fully random, and Bootstrap_num_steps sweeps are used instead of
                        num_steps -- see Parallel_Tempering_Partition's docstring for the
                        rationale (cold chains benefit most from a warm start; hot chains exist
                        to explore independently and shouldn't be biased toward data)
                    --> self.PT_chain_cache persists across epochs AND across separate calls to
                        this method; call self.Reset_PT_Chain_Cache(partition_index) to clear it
                        (e.g. after a large weight change) if the cached chains might be stale
                        -- doing so re-triggers the bootstrap (anchoring + extra steps) for
                        every example on its next encounter

        - Gradient = anti_Hebbian - Hebbian, masked by self.fundamental_connectivity
                    --> this is (model - data), not the more familiar (data - model),
                        because Energy_Mutant_Array_light has no leading minus sign; 
                        Purely because Stat mech sign convention in energy and probability
                        a proper gradient ASCENT step (self.Weights += lr * Dw_full)
                    --> only the within-partition (self) block is symmetrized;
                        the interact block (influence from the previous partition) is NOT symmetric
        - NaN check: if W_error or B_error is NaN, skip the update entirely (no step taken)
        - Adaptive (RMSProp-like) learning rates used if self.Adaptive == True, else flat self.learning_rate
                    --> self.Adaptive, self.learning_rate, self.decay_rate, self.epsilon default
                        lazily (with a printed notice) if not already set externally;
                        self.squared_grad_weights/bias and the lr caches are per-partition
                        dicts, since gradient shape varies by partition

        - Output:
            - W_error = mean absolute weight gradient (normalized by connectivity)
            - B_error = mean absolute bias gradient
            - Recons_error = multiplicity-weighted average (over the subsample) of the minimum
                              reconstruction distance between the true state and any sampled
                              chain, RESTRICTED to the non-anchored ("probe") slots -- the
                              hotter half of the ladder that Parallel_Tempering_Partition never
                              seeds at data (self.PT_Anchored_Slot_Indices), so a low value here
                              reflects genuine model-driven proximity to data, not a forced start
            - PLL_error = multiplicity-weighted average pseudo-negative-log-likelihood of the
                              true data under the CURRENT weights (self.Pseudo_LogLikelihood_Local),
                              computed over ALL unique examples with no PT/sampling involved --
                              UNCONFOUNDED by chain initialization, anchoring, or num_steps;
                              lower is better
        """

        if Bootstrap_num_steps is None:
            Bootstrap_num_steps = num_steps * 3

        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]
        Connectivity_part = self.fundamental_connectivity[partition_index]

        #### ---- Reduce to partition-local unique examples + their local multiplicity ----
        ## Two full sequences that agree on this partition's D_prev and Data_self are
        ## locally identical for this partition's energy function, even if they differ
        ## elsewhere. Collapsing to unique local examples avoids redundant PT runs and
        ## weights each local context by how often it actually occurs.
        Data_self_unique, Data_interact_unique, Multiplicity_unique, _ = self.Data_Partition_args_and_multiplicity(State_Array, partition_index)

        Diff = Data_interact_unique.shape[1] - Data_self_unique.shape[1]
        D_prev_unique = Data_interact_unique[:, :Diff]

        Num_unique = len(Data_self_unique)

        ## weight each unique local example by its frequency in State_Array,
        ## normalized so weights sum to 1 across the full (non-subsampled) set
        P_eg = Multiplicity_unique / len(State_Array)

        #### ------------------- Positive (Hebbian) phase ------------------------
        Scaled_examples = Data_self_unique * P_eg[:, np.newaxis]

        db_hebb = np.sum(Scaled_examples, axis=0)
        dw_self_hebb = Scaled_examples.T @ Data_self_unique
        dw_interact_hebb = Scaled_examples.T @ D_prev_unique

        ##### --------------------- End Hebbian Phase ------------------------------

        #### ------------ Pseudo-likelihood diagnostic (unconfounded by PT) -----------
        ## Cheap: no Gibbs/PT needed. Computed over ALL unique examples (like the Hebbian
        ## phase), weighted by P_eg -- so this is a comprehensive, low-variance signal of
        ## how much probability the CURRENT weights place on the true data, completely
        ## independent of chain initialization, anchoring, or num_steps.
        PLL_error = 0.0
        for i in range(Num_unique):
            PLL_error += P_eg[i] * self.Pseudo_LogLikelihood_Local(
                Data_self_unique[i], D_prev_unique[i], partition_index, temperature=1
            )

        #### ------------ Anti-Hebbian Phase with Parallel Tempering -----------

        dw_self_anti_hebb = 0
        dw_interact_anti_hebb = 0
        db_anti_hebb = 0

        ### Pick a subset of unique local examples — resampled every call.
        ### Sampled UNIFORMLY over unique examples for coverage; each picked example's
        ### contribution is then importance-weighted by its (renormalized) multiplicity,
        ### consistent with how the Hebbian phase is weighted.
        Num_pick = int(Num_unique * batch_fraction)
        Num_pick = np.min((Num_pick, Num_unique))
        Picked_indices = np.random.choice(np.arange(0, Num_unique), Num_pick, replace=False)

        Data_self_subsample = Data_self_unique[Picked_indices]
        D_prev_subsample = D_prev_unique[Picked_indices]

        P_eg_subsample = P_eg[Picked_indices]
        P_eg_subsample = P_eg_subsample / np.sum(P_eg_subsample)   # renormalize within the subsample

        Recons_error = 0

        ## Recons_error should reflect samples the model genuinely produced, not slots we
        ## forced to start at data. Exclude the anchored (coldest-half) slots from the min --
        ## same slot set Parallel_Tempering_Partition itself uses for anchoring, so this stays
        ## correct even as that logic evolves.
        anchored_slots = set(self.PT_Anchored_Slot_Indices(Temperature_list).tolist())
        Probe_slot_indices = np.array([i for i in range(num_chains) if i not in anchored_slots])
        if len(Probe_slot_indices) == 0:
            Probe_slot_indices = np.arange(num_chains)   # fallback if every slot is anchored

        ## persistent chain cache: keyed by partition, then by a hash of [D_prev | Data_self]
        ## for that example, so the same example's chains survive across epochs instead of
        ## being re-mixed from scratch every call (persistent CD)
        if not hasattr(self, 'PT_chain_cache'):
            self.PT_chain_cache = {}
        if partition_index not in self.PT_chain_cache:
            self.PT_chain_cache[partition_index] = {}
        Partition_chain_cache = self.PT_chain_cache[partition_index]

        for idx in tqdm(range(Num_pick)):

            weight = P_eg_subsample[idx]

            Data_self_true = Data_self_subsample[idx]
            D_prev_this = D_prev_subsample[idx]

            example_key = self.Hash_Value_given_data(np.concatenate([D_prev_this, Data_self_true]))
            Cached_chains = Partition_chain_cache.get(example_key, None)

            ## bootstrap (no cache yet) gets more sweeps, since anchored/random chains
            ## start further from equilibrium than a persisted chain does
            steps_this_call = num_steps if Cached_chains is not None else Bootstrap_num_steps

            Chains, _ = self.Parallel_Tempering_Partition(
                partition_index, Temperature_list, num_steps=steps_this_call, Step_Split=Step_Split,
                D_prev=D_prev_this, Initial_Chains=Cached_chains, Anchor_State=Data_self_true,
                Anchor_Max_Noise=Anchor_Noise)

            Partition_chain_cache[example_key] = Chains.copy()

            Calculated_energy_of_chain = self.Energy_Mutant_Array_light(Weight_part, Bias_part, Chains, D_prev_this)
            Probability_chain = self.Softmax_probability_stable(Calculated_energy_of_chain, sign=-1)

            Scaled_chain_data = Chains * Probability_chain.reshape(-1, 1)

            Abs_difference = np.sum(np.abs(Data_self_true - Chains), axis=1) * 0.5
            Recons_error += weight * np.min(Abs_difference[Probe_slot_indices])

            dw_self_anti_hebb += weight * (Scaled_chain_data.T @ Chains)
            db_anti_hebb += weight * np.sum(Scaled_chain_data, axis=0)
            dw_interact_anti_hebb += weight * (Scaled_chain_data.T @ np.tile(D_prev_this, (num_chains, 1)))

        ### ------------------------- End Anti-Hebbian Phase ------------------------

        Dw_self_full = (dw_self_anti_hebb - dw_self_hebb)
        Dw_self_full = (Dw_self_full + Dw_self_full.T) / 2   # symmetrize only within-partition block

        Dw_interact_full = (dw_interact_anti_hebb - dw_interact_hebb)   # not symmetrized

        Dw_full = np.hstack([Dw_interact_full, Dw_self_full]) * Connectivity_part
        Db_full = (db_anti_hebb - db_hebb)

        W_error = np.sum(np.abs(Dw_full)) / np.sum(Connectivity_part)
        B_error = np.mean(np.abs(Db_full))

        if math.isnan(W_error) or math.isnan(B_error):
            print("NaN encountered!! No Update")
            Dw_full = 0
            Db_full = 0

        #### -------- Lazy defaults for training hyperparameters --------
        ## these are expected to be set externally, but if they weren't
        ## we can fall back to sensible defaults instead of raising AttributeError
        if not hasattr(self, 'Adaptive'):
            print("self.Adaptive not set -- defaulting to False (flat learning rate)")
            self.Adaptive = False

        if not hasattr(self, 'learning_rate'):
            print("self.learning_rate not set -- defaulting to 0.01")
            self.learning_rate = 0.01

        if self.Adaptive == True:
            if not hasattr(self, 'decay_rate'):
                print("self.decay_rate not set -- defaulting to 0.9")
                self.decay_rate = 0.9

            if not hasattr(self, 'epsilon'):
                print("self.epsilon not set -- defaulting to 1e-8")
                self.epsilon = 1e-8

            ## squared_grad_weights/bias and the lr caches are keyed PER PARTITION,
            ## since Dw_full / Db_full shape varies by partition (unlike a single global array)
            if not hasattr(self, 'squared_grad_weights'):
                self.squared_grad_weights = {}
            if not hasattr(self, 'squared_grad_bias'):
                self.squared_grad_bias = {}
            if not hasattr(self, 'weight_lr_cache'):
                self.weight_lr_cache = {}
            if not hasattr(self, 'bias_lr_cache'):
                self.bias_lr_cache = {}

            if partition_index not in self.squared_grad_weights:
                self.squared_grad_weights[partition_index] = np.zeros_like(Weight_part)
            if partition_index not in self.squared_grad_bias:
                self.squared_grad_bias[partition_index] = np.zeros_like(Bias_part)

        if self.Adaptive == True:
            self.squared_grad_weights[partition_index] = (
                self.decay_rate * self.squared_grad_weights[partition_index]
                + (1 - self.decay_rate) * (Dw_full ** 2)
            )
            self.squared_grad_bias[partition_index] = (
                self.decay_rate * self.squared_grad_bias[partition_index]
                + (1 - self.decay_rate) * (Db_full ** 2)
            )

            adaptive_lr_weights = self.learning_rate / (np.sqrt(self.squared_grad_weights[partition_index]) + self.epsilon)
            adaptive_lr_bias = self.learning_rate / (np.sqrt(self.squared_grad_bias[partition_index]) + self.epsilon)

            self.weight_lr_cache[partition_index] = adaptive_lr_weights
            self.bias_lr_cache[partition_index] = adaptive_lr_bias
        else:
            adaptive_lr_weights = self.learning_rate
            adaptive_lr_bias = self.learning_rate

        self.Weights[partition_index] += adaptive_lr_weights * Dw_full
        self.Bias[partition_index] += adaptive_lr_bias * Db_full

        return W_error, B_error, Recons_error, PLL_error


    def Train_Contrastive_Divergence(
        self,
        Data,
        partition_index,
        temp_list,
        num_epochs,
        learning_rate=0.01,
        decay_rate=0.9,
        epsilon=1e-8,
        Adaptive=False,
        num_steps=10,
        Step_Split=0.5,
        batch_fraction=1.0,
        reset_chains=False,
        Bootstrap_num_steps=None,
        Anchor_Noise=0.3,
        track_best_checkpoint=True):

        """Runs Contrastive_Divergence_Partition repeatedly (one call per epoch) for a
        single partition, and collects the error trace over training:
        Given:
        -  Data = full unfurled data (any batch, duplicates allowed), passed straight
                  through to Contrastive_Divergence_Partition as State_Array each epoch
        -  partition_index = the partition being trained
        -  temp_list = temperature ladder for Parallel Tempering
                    --> num_chains is derived as len(temp_list), NOT taken as a separate
                        argument, since the two must always match (Parallel_Tempering_Partition
                        makes exactly one chain per temperature)
        -  num_epochs = number of CD updates to perform (one per epoch)
        -  learning_rate, decay_rate, epsilon, Adaptive = learning attributes,
                    set onto self ONCE before training begins
                    --> this OVERRIDES any existing self.learning_rate / self.Adaptive / etc.,
                        it does not just lazily default them like Contrastive_Divergence_Partition
                        does on its own
                    --> switching Adaptive True->False (or vice-versa) mid-training by calling
                        this again is fine; the per-partition squared-grad state is preserved
                        across calls (only reset if you delete self.squared_grad_weights etc.
                        yourself)
        -  num_steps, Step_Split, batch_fraction = passed straight through to
                    Contrastive_Divergence_Partition every epoch (same value used throughout)
        -  reset_chains = if True, clears self.PT_chain_cache for this partition before
                    training starts (self.Reset_PT_Chain_Cache(partition_index))
                    --> the negative-phase PT chains persist across epochs by default
                        (persistent CD), which is normally what you want; set this True if
                        you're resuming training after a large/unusual weight change and
                        don't trust the old cached chains to still be near equilibrium
                    --> only affects the FIRST epoch of THIS call: every example's chains
                        still persist normally across the num_epochs that follow
                    --> re-triggers graded anchoring + Bootstrap_num_steps for every example
                        on its next encounter, since it clears their cache entries too
        -  Bootstrap_num_steps, Anchor_Noise = passed straight through to
                    Contrastive_Divergence_Partition every epoch; only affect examples on
                    their first-ever (or post-reset) encounter -- see that method's docstring
        -  track_best_checkpoint = if True (default), after each epoch compares this epoch's
                    PLL_error against the best seen so far for this partition and, if it
                    improved, snapshots self.Weights[partition_index] / self.Bias[partition_index]
                    (self.Update_Best_Checkpoint) -- since training here is noisy (subsampled
                    negative phase, PT sampling variance, adaptive LR overshoot), the FINAL
                    epoch's weights aren't guaranteed to be the best ones seen; call
                    self.Restore_Best_Weights(partition_index) afterward to load the
                    checkpoint back in. This is checkpointed on TRAINING PLL, not held-out
                    PLL -- see Update_Best_Checkpoint's docstring

        - Returns
            - Error_trace = array of shape (num_epochs, 4),
                    columns = [W_error, B_error, Recons_error, PLL_error]
                    --> Error_trace[:,0] = W_error over epochs
                    --> Error_trace[:,1] = B_error over epochs
                    --> Error_trace[:,2] = Recons_error over epochs (PT-based, probe-slots only)
                    --> Error_trace[:,3] = PLL_error over epochs (PT-free, unconfounded; lower better)
        """

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.Adaptive = Adaptive

        if reset_chains:
            self.Reset_PT_Chain_Cache(partition_index)

        num_chains = len(temp_list)

        Error_trace = np.zeros((num_epochs, 4))

        for epoch in range(num_epochs):

            W_error, B_error, Recons_error, PLL_error = self.Contrastive_Divergence_Partition(
                partition_index,
                Data,
                temp_list,
                num_chains,
                num_steps=num_steps,
                Bootstrap_num_steps=Bootstrap_num_steps,
                Anchor_Noise=Anchor_Noise,
                Step_Split=Step_Split,
                batch_fraction=batch_fraction,
            )

            Error_trace[epoch, 0] = W_error
            Error_trace[epoch, 1] = B_error
            Error_trace[epoch, 2] = Recons_error
            Error_trace[epoch, 3] = PLL_error

            improved_tag = ""
            if track_best_checkpoint:
                if self.Update_Best_Checkpoint(partition_index, PLL_error):
                    improved_tag = "  [new best -- checkpointed]"

            print(f"Epoch {epoch+1}/{num_epochs} -- W_error={W_error:.6g}, B_error={B_error:.6g}, Recons_error={Recons_error:.6g}, PLL_error={PLL_error:.6g}{improved_tag}")

        return Error_trace