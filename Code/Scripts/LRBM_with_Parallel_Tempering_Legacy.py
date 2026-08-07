#Author: Bipul Pandey
# Date: June 20 2025
# Contains Layer-Restricted Boltzmann Machine
    # - with parallel tempering
    # - adaptive learning rate


import numpy as np
import matplotlib.pyplot as plt
import math 

from tqdm import tqdm
import pandas as pd

from Boltzmann_Function_gen_2 import *

from scipy.stats import pearsonr
from scipy.stats import kstest






############# K-means clustering Functions ############
## Helpers


def Manhattan_dist(Centers, P2):
    
    Difference = Centers-P2
    
    Dist = np.abs((Difference))
    return Dist




def kmeans_details_verbose_manhattan(Points, Centers, metric = "Euclidean"):
    Distance_min =[]
    Cluster_id=[]
    Avg_dist = []

    Num_clusters = len(Centers)

    for p in Points:
        Distances = Manhattan_dist(Centers, p)

        minimal_distance = float(np.min(Distances))

        Avg_distance = (np.mean(Distances))

        Cluster_assignment = int(np.argmin(Distances))

        Distance_min.append(minimal_distance)
        Cluster_id.append(Cluster_assignment)
        Avg_dist.append(Avg_distance)

    return np.array(Avg_dist), np.array(Cluster_id), np.array(Distance_min)




def Euc_dist_point(P1, P2):
    
    Difference = P1-P2
    
    Dist = np.sqrt(np.sum(Difference**2))
    return Dist

def Euc_dist(Centers, P2):
    
    Difference = Centers-P2
    
    Dist = np.sqrt(np.sum(Difference**2, axis = 1))
    return Dist




def kmeans_details_verbose(Points, Centers, metric = "Euclidean"):
    Distance_min =[]
    Cluster_id=[]
    Avg_dist = []

    Num_clusters = len(Centers)

    for p in Points:
        if metric == "Euclidean":
            ## default
            Distances = Euc_dist(Centers, p)
        else:
            Distances = Manhattan_dist(Centers, p)

        minimal_distance = float(np.min(Distances))

        Avg_distance = (np.mean(np.sort(Distances)))

        Cluster_assignment = int(np.argmin(Distances))

        Distance_min.append(minimal_distance)
        Cluster_id.append(Cluster_assignment)
        Avg_dist.append(Avg_distance)

    return np.array(Avg_dist), np.array(Cluster_id), np.array(Distance_min)


def kmeans_details_verbose_1(Points, Centers):
    Distance_array =[]
    Cluster_id=[]
    Avg_dist = []

    Num_clusters = len(Centers)

    for p in Points:
        Distances = Euc_dist(Centers, p)

        Distance_array.append(Distances)
        Cluster_assignment = int(np.argmin(Distances))
        Cluster_id.append(Cluster_assignment)
        

    return  np.array(Cluster_id), np.array(Distance_array)



def Coefficient_to_vector(Coefficient_list, Vector_list):
    Z = np.zeros(len(Coefficient_list))
    for i,c in enumerate(Coefficient_list):
        Z+= c * Vector_list[i]
    return Z


############# End K-means clustering Functions ############




###### Helpers #####

def make_unfurled_state(State, Ordered_alphabet_list):
    """----- Compact State unfurler ----
    --- takes compact state in alphabet notation

    --- takes alphabet list in order (alphabet list size = n)
    --- takes subsequent alphabet into one hot encoded basis
     """

    num_features = len(State)
    num_alphabet = len(Ordered_alphabet_list)

    length_unfurled = num_features * num_alphabet

    unfurled_state = []

    for value in State:

        alphabet_unfurled = np.zeros(num_alphabet)

        arg_where_one = np.argwhere(Ordered_alphabet_list ==value)
        alphabet_unfurled[arg_where_one] =1
        unfurled_state.append(alphabet_unfurled)

    unfurled_state = np.ravel(np.array(unfurled_state))

    # check if unfurled length matches expected length:
    if len(unfurled_state) != length_unfurled:
        print("Unfurled state length does not match expected length")
        print("Expected:", length_unfurled)
        print("Actual:", len(unfurled_state))
        return

    return unfurled_state


def make_compact_state(unfurled_state, ordered_alphabet_list):

    """----- Unfurled State Compactor ----
    --- takes one hot encoded unfurled state
    --- takes alphabet list in order (alphabet list size = n)
    --- takes subsequent 'n' elements and converts it into corresponding alphabet    
     """
    
    num_alphabet = len(ordered_alphabet_list)
    num_unfurled_features = len(unfurled_state)

    State = []

    for i in range(0, num_unfurled_features, num_alphabet):

        start = i
        stop = i +num_alphabet

        read_state = unfurled_state[start:stop]

        if np.sum(read_state)!=1:
            print("State cannot have two alphabets at the same location. Check:", start, stop)
            return 

        arg_true = np.ravel(np.argwhere(read_state ==1))

        State.append(ordered_alphabet_list[arg_true])

    State = np.array(State)

    return State



def Von_Neumann_Entropy_per_residue(Data_unfurled, Alphabet_size = 21):

    """Calculates Von Neumann Entropy for each site for the unfurled dataset:

    Inputs:
        - Unfurled Data Set (Each site is represented as one hot encoded vector of length =Alphabet size)

        - Alphabet size: Total number of Alphabets

    Output:
        -Von Neumann entropy (Base 2) for each site
        - Effective dimension = Base ^ (Von Neumann entropy)
        """

    ### First thing, check if alphabet size is correct:
    Length_one_hot_encoded = np.shape(Data_unfurled)[1],
    if np.mod(Length_one_hot_encoded, Alphabet_size)>0:
        print("Data Size and Alphabet size mismatch")
        return None
     
    VN_site = []

    for i in range(0, np.shape(Data_unfurled)[1], Alphabet_size):
        start = i
        end = i+ Alphabet_size

        Data = Data_unfurled[:, start:end]
        _,si,_ = np.linalg.svd(Data)

        Eigvals = si**2/np.sum(si**2)
        Eigvals_usable = Eigvals[np.ravel(np.argwhere(Eigvals>0.0))]

        VN_entropy_site_i = np.sum(-1*np.log2(Eigvals_usable) * Eigvals_usable)
    
        VN_site.append(VN_entropy_site_i)

    return np.array(VN_site)

    


def Softmax_probability_stable(Energy_list, sign = 1, temperature = 1):

    """----- stable softmax function that doesn't overflow ----
    --- if sign = 1 --> softmax of Energy_list (default)
    --- if sign = -1 --> softmin of Energy_list

    --- temperature scales energy by division (default temperature = 1)
    
     """
    Energy_array = np.array(Energy_list) * sign

    max_value = np.max(Energy_array)

    Energy_array_scaled = (Energy_array -max_value)/temperature

    likelihood_energy = np.exp(Energy_array_scaled)

    normalization = np.sum(likelihood_energy)

    Softmax = likelihood_energy/normalization

    return Softmax





def make_corrupted_data(Train_data, Ordered_alphabet_list, corruption_rate = 0.1):
    """----- Corruption of the data ----
    --- takes in the training data (compact) and corrupts it by a given rate
    --- returns the corrupted data
     """

    num_features = len(Train_data)
    num_alphabet = len(Ordered_alphabet_list)

    length_unfurled = num_features * num_alphabet

    corrupted_data = Train_data.copy()

    for i in range(num_features):
        if np.random.rand() < corruption_rate:
            corrupted_data[i] = np.random.choice(Ordered_alphabet_list, 1)[0]

    return corrupted_data



def Checker(seq, Train_data_unfurled):
    """Checks to see which sequence form the training data is closest to the given sequence (seq)
    
    - Input:
    seq= sequence in question
    Train_data_unfurled = Training dataset in the same basis
    
    -Output:
    - Status= True/False : Is the sequence in the training set
    - Max_val = maximul similarity value
    - arg_max = argument for the closest match in training data
    
    """

    Checker = np.sum(seq == Train_data_unfurled, axis=1)
    max_val = np.max(Checker)
    arg_max = np.argmax(Checker)

    if max_val == np.sum(Train_data_unfurled[0] == Train_data_unfurled[0]):
        Status = True
    else:
        Status = False
    return Status, max_val, arg_max




def Partititon_arg(Partition, num_letters):

    """Given a partition and number of letters,
    ---- Input ----
    - Partition : a [0,1] vector of length L (length of compact sequence) where
                    0--> that site is not in this partiton
                    1--> the site is in this partition

    - num_letters : number of internal degrees of freedom for each site (20 +1 gap = 21 for proteins)

    
    ----
    It gives all the arguments in the unfurled list that belong to this partition
    
    """ 

    Args_raw = np.ravel(np.argwhere(Partition ==1).reshape(-1,1))

    Args_unfurled = []
    for elem in Args_raw:

        start = elem * num_letters
        end = start + num_letters

        args_elem = np.arange(start, end, 1)

        Args_unfurled.append(args_elem)
    return np.concatenate(Args_unfurled)
####---------------

class Layered_Multimodal_Boltzmann_Machine:

    def __init__(self, alphabet_list, Current_layer_partition,  Previous_layer_partition=None , learning_rate = 0.001, temperature = 1):
        """
        Initialize a multimodal Boltzmann Machine
        
        Parameters:
        -----------
        num_unfurled_features : int
            -- Number of features (sites/position) in the actual data
        num_states_per_unit : int
        alphabet_list: array
            -- possible states for each position
        learning_rate : float
            -- Learning rate for gradient updates
        """

        ############# number of alphabets #########

        self.alphabet_lists = alphabet_list
        self.num_alphabet = len(alphabet_list)


        ########## current layer properties

        self.num_unfurled_features  = int(np.sum(Current_layer_partition)) * self.num_alphabet

        self.num_features = self.num_unfurled_features// self.num_alphabet
        
        
        ### check for error mismatch:
        if self.num_alphabet* self.num_features != self.num_unfurled_features:
            print("Check dimensions of alphabet list and unfurled state. Dimensional mis-match")


        ##### BM properties #####################
        
        self.learning_rate = learning_rate
        self.temperature= temperature

        ##########################################


        ##### initializing weights and  bias for current layer
        fundamental_connectivity = np.ones((self.num_unfurled_features, self.num_unfurled_features))


        for i in range(0, self.num_unfurled_features, len(alphabet_list)):
            start = i 
            end = i + len(alphabet_list)
            fundamental_connectivity[start:end, start:end]=0

        self.fundamental_connectivity = fundamental_connectivity

        ### Current_layer - Current_layer connectivity (Weight)
        Weight_random =np.random.normal(loc = 0, scale = learning_rate/100, size =(self.num_unfurled_features, self.num_unfurled_features))
        Weight_random *= fundamental_connectivity

        self.weights = Weight_random

        ### Current_layer bias
        self.bias = np.random.normal(loc = 0, scale = learning_rate/100, size = self.num_unfurled_features)

        ###########-------Previous layer properties--------------------##############

        if np.sum(Previous_layer_partition)==None:
            ### no previous layer
            ### or no influence--> Independent layers
            self.Previous_layer_size = 0
            self.Influence = False
            self.Influence_matrix = 0
            

        else:
            self.Previous_layer_size = int(np.sum(Previous_layer_partition)) * self.num_alphabet
            self.Influence = True
            self.Influence_matrix = np.random.normal(loc = 0, scale = learning_rate/100, size  =(self.num_unfurled_features, self.Previous_layer_size))

        #############################################################

    def set_weight(self, weight_matrix):
        fundamental_connection = self.fundamental_connectivity
        ###check shape before setting:
        D1, D2 = np.shape(weight_matrix)


        if D1!=self.num_unfurled_features or D2!=self.num_unfurled_features:
            print("Dimensional Mismatch between Boltzmann Machine and this weight matrix")
        else:
            self.weights = weight_matrix*fundamental_connection


    def set_bias(self, bias_vector):

        ###check shape before setting:
        D1 = len(bias_vector)
        if D1!=self.num_unfurled_features:
            print("Mismatch in the Dimension of Boltzmann Machine and this Bias vector")
        else:
            self.bias = bias_vector



    def get_weight_bias(self):

        W = self.weights
        B = self.bias

        return W, B



    def Set_previous_layer_parameter(self,Previous_layer_partition):

        if np.sum(Previous_layer_partition) >0:
            self.Previous_layer_size = np.sum(Previous_layer_partition) * self.num_alphabet

            self.Influence = True
        
            self.Influence_matrix = np.random.normal(0, self.learning_rate/100, size=(self.num_unfurled_features,self.Previous_layer_size ))



    def Self_Energy_array(self, Unfurled_Current_layer_State_array):

        ### The energy contribution because of the interactions within the current layer
        ### Hence we call this the self energy
        Weights = self.weights
        Bias = self.bias

        if len(Unfurled_Current_layer_State_array)>1:
            Energy_all = np.sum(-((Unfurled_Current_layer_State_array@Weights) + Bias) * Unfurled_Current_layer_State_array, axis =1)

        else:
            ### there is a single state
            Energy_all = -((Unfurled_Current_layer_State_array@Weights) + Bias) * Unfurled_Current_layer_State_array

        return Energy_all
    

    def Influence_bias(self,Unfurled_Previous_layer_State_array):
        Influence_bias_contrib=0
        if self.Influence==True:
            
            Influence_bias_contrib = Unfurled_Previous_layer_State_array@ self.Influence_matrix.T
            
        return Influence_bias_contrib
    

    
    def Influence_Energy(self, Unfurled_Previous_layer_State_array, Unfurled_Current_layer_State_array):
        ## compute influence
        Influence_energy = 0

        if self.Influence==True:
            Influence_of_prev_layer = self.Influence_bias(Unfurled_Previous_layer_State_array)
            Influence_energy = -1* np.sum(Unfurled_Current_layer_State_array*Influence_of_prev_layer, axis=1)

        return Influence_energy


    def Energy_state(self, Unfurled_Previous_layer_State_array, Unfurled_Current_layer_State_array):

        """ __________________________________________________________________
        Computes Energy of a single state given:
        - Unfurled_Previous_layer_State_array --> data for that state projected in previous layer
        - Unfurled_Current_layer_State_array--> data for that state projected in current layer
        ________________________________
        ---!---  If prev_layer_size =0 ---!--- :
        - Only Data_current_layer used 
        - Influence becomes 0
        _______________________________________________________________________
        """

        ## the self energy part
        Self_energy = self.Self_Energy_array(Unfurled_Current_layer_State_array)

        Total_Energy_of_state = Self_energy
        
        ## the influence energy part
        if self.Influence==True:
            if len(Unfurled_Previous_layer_State_array)>1:
                Influence_energy = self.Influence_Energy(Unfurled_Previous_layer_State_array, Unfurled_Current_layer_State_array)
            else:
                ## there is a single state
                Influence_energy = self.Influence_Energy(Unfurled_Previous_layer_State_array, Unfurled_Current_layer_State_array)[0]

            Total_Energy_of_state += Influence_energy

        return Total_Energy_of_state



    def Softmax_probability_stable(self, Energy_list, sign = 1 , temperature = None):

        """----- stable softmax function that doesn't overflow ----
        --- if sign = 1 --> softmax of Energy_list (default)
        --- if sign = -1 --> softmin of Energy_list

        --- temperature scales energy by division (default temperature = 1)
        
        """

        ### if no temperature is given, use default temperature (which is set to 1).
        if temperature==None:
            temperature = self.temperature

        Energy_array = np.array(Energy_list) * sign

        max_value = np.max(Energy_array)

        Energy_array_scaled = (Energy_array - max_value)/temperature

        likelihood_energy = np.exp(Energy_array_scaled)

        normalization = np.sum(likelihood_energy)

        Softmax = likelihood_energy/normalization

        return Softmax
    



    def Activation_probability_full(self, Unfurled_Previous_layer_State, Unfurled_Current_layer_State, temperature = None):
            """Gives the activation probability for the each position of the entire state
            - Full partition state (single mutant space) is created
            - Energy for each single mutant is constructed
            - For each position, activation probability is found using softmin
            """


            Length = self.num_unfurled_features
            

            if temperature==None:
                temperature = self.temperature

            ####################################
            fundamental_connection = self.fundamental_connectivity

            num_alphabets = self.num_alphabet

            ### construct single change full partition space for this configuration:
            Full_partition = fundamental_connection * Unfurled_Current_layer_State
            ### the diagonal of full partition must be 1 since we are sweeping through feature and alphabet.
            Full_partition+= np.eye(Length) 
        

            ### Rather than computing the full energy using the function,
            ### I will compute them here by calculating only the terms that do not cancel out for this partition.

            Prev_full = np.array([Unfurled_Previous_layer_State for _ in range(0, len(Full_partition))])
            Energy_full_partition = self.Energy_state(Prev_full, Full_partition)

            Probability_array = np.zeros_like(Energy_full_partition)

            for i in range(0, len(Energy_full_partition), num_alphabets):
    
                start = i 
                end = i + num_alphabets

                E_frag = Energy_full_partition[start:end]

                Probability_array[start:end] = self.Softmax_probability_stable(E_frag, sign = -1, temperature= temperature)

            return Probability_array
    


    def Activation_probability_given_index(self, Unfurled_Previous_layer_State, Unfurled_Current_layer_State, temperature = None, indices_to_change=None):
            """Gives the activation probability for
              selected degrees of freedom (selected sites through given index) in the state"""
            


            Length = self.num_unfurled_features
    
            if None in np.array(indices_to_change):
                ## compute activation energy for all sites if indices are not given
                indices_to_change =range(0,Length, num_alphabets)
        
            

            if temperature==None:
                temperature = self.temperature

            ####################################
            fundamental_connection = self.fundamental_connectivity

            num_alphabets = self.num_alphabet

            ### construct single change full partition space for this configuration:
            Full_partition = fundamental_connection * Unfurled_Current_layer_State

            ### the diagonal of full partition must be 1 since we are sweeping through feature and alphabet.
            #Full_partition+= np.eye(Length) 
            # For now, we can leave this diagonal as there is no self connection.
        

            ### Rather than computing the full energy using the function,
            ### I will compute them here by calculating only the terms that do not cancel out for this partition.

            Weights = self.weights
            Bias = 0
            Bias += self.bias

            if self.Influence == True:
                # then the influence of previous layer will also have a bias contribution
                Influence_bias = self.Influence_bias(Unfurled_Previous_layer_State)
                Bias+= Influence_bias
            ## everything from here on remains the same

            Probability_array = np.zeros(Length)

            for i in indices_to_change:
                ### we will only take into account the piece of energy that is different 
                ### when we switch alphabet within a site.

                start = i * num_alphabets
                end = start +num_alphabets

                ### the following vector contains the site in question as well as all the other sites
                ### Fundamental connectivity makes sure that we can slice the weight matrix as follows without running
                ### into issues of self connectivity.
                ## For any site, all we care about are what are in all other positions besides that site

                vec_i = Full_partition[start]  

                ###  Weights_i_all = effect of site in question on all other sites
                ### Weights_all_i = effect of all other sites on site in question
                Weights_i_all = Weights[:,start:end]

                Weights_all_i = Weights[start:end]

                E_i_all = vec_i @ Weights_i_all      #< The effect of having a particular site's muatnt on all
                E_all_i = (Weights_all_i @ vec_i.T)   #< The effect of all positions on a partitcular mutant at a site

                E_site = -1*(E_i_all+E_all_i + Bias[start:end])
                
                ### rather than softmax I want a softmin 
                ## hence sign = -1 in softmax
                Prob_site = self.Softmax_probability_stable(E_site, sign=-1, temperature=temperature)
                Probability_array[start:end]= Prob_site

            return Probability_array
    



    def Forward_Gibbs_pass_partitioned(self, Unfurled_Previous_layer_State, Unfurled_Current_layer_State, temperature , partition = 0.5):

        #### rather than changing all sites, we will only change a fraction of the sites

        Length = self.num_features
        num_alphabet = self.num_alphabet
  
        Num_sites_to_change = np.max((int(Length*partition),1))    #<== At least I will change a single site, partition=0

        ### get the indices of the sites to change
        indices_to_change = np.sort(np.random.choice(range(0, Length), Num_sites_to_change, replace=False))
        #indices_to_keep = np.setdiff1d(range(0, Length), indices_to_change)
        temp = temperature
        ### get the activation probability for the indices selected above for the state
        Activation_prob = self.Activation_probability_given_index(Unfurled_Previous_layer_State=Unfurled_Previous_layer_State, Unfurled_Current_layer_State=Unfurled_Current_layer_State, temperature=temperature, indices_to_change=indices_to_change)
        
        ### Initialize the Next State as as copy of the current state
        ### we will keep the indices that are not changing (by not operating on them)
        Next_state = Unfurled_Current_layer_State.copy()
        #Next_state[indices_to_keep] = State_unfurled[indices_to_keep]

        for index in indices_to_change:

            start = index*num_alphabet
            stop = start + num_alphabet

            Prob_i = Activation_prob[start:stop]

            Picked_state = np.random.choice(range(0,num_alphabet), p=Prob_i)
            ## blank out the previous state at this site
            Next_state[start:stop] = 0
            
            ## replace it with the new state
            Next_state [start+Picked_state]=1

        return Next_state
    

    def gibbs_sampling_partitioned(self, Unfurled_Previous_layer_State, Unfurled_Current_layer_State, num_steps, partition = 0.5, temperature=None):

        if None in np.array(temperature):
            temperature = self.temperature

        State = Unfurled_Current_layer_State.copy()

        for i in range(0, num_steps):
            State = self.Forward_Gibbs_pass_partitioned(Unfurled_Previous_layer_State, State, temperature , partition)
            
        return State
    


    def Random_state_unfurled(self):
        num_features = self.num_features
        num_alphabet = self.num_alphabet
        Ordered_alphabet_list = self.alphabet_lists

        State = np.random.choice(Ordered_alphabet_list,num_features)
        length_unfurled = num_features * num_alphabet

        unfurled_state = []

        for value in State:

            alphabet_unfurled = np.zeros(num_alphabet)

            arg_where_one = np.argwhere(Ordered_alphabet_list ==value)
            
            alphabet_unfurled[arg_where_one] =1
            unfurled_state.append(alphabet_unfurled)

        unfurled_state = np.ravel(np.array(unfurled_state))

        return unfurled_state
    
    

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
        


        if type==1:
            ### Inverse-linear temperature scheme
            ### linear steps in beta (the inverse temperature)

            beta_low_temp = 1/min_temp
            beta_high_temp = 1/max_temp

            beta_steps = np.linspace(beta_low_temp, beta_high_temp, num_chains)

            Temp_steps = 1/beta_steps


        if type ==2:
            ### Geometric temperature steps scheme

            ratio = (max_temp/min_temp)**(1/(num_chains-1))
            r_step = np.arange(0, num_chains)

            Temp_steps = min_temp *(ratio **r_step)


        if type ==3:
            ### linear temperature steps scheme
            Temp_steps =  np.linspace(min_temp, max_temp, num_chains)


        if type ==4:
            #all temperatures same
            ## here all we are doing is exploring the space with walkers of same energy
            Temp_steps = np.ones(shape = num_chains)

        return Temp_steps
    



    def Parallel_tempering(self, Unfurled_Previous_layer_State,  temperature_list, Unfurled_Current_layer_State=None, num_steps = 10, partition = 0.5):

        """Parallel tempering for the Boltzmann machine:
        Given:
        -  Unfurled_Previous_layer_State = Data for previous layer for this state
        -  temperature list --> (the number of chains = length of this list)
        -  Unfurled_Current_layer_State-= Current layer for this state (if None [default], start with random state)
        -  num_steps = number of steps to take  (default =10)
        -  partition = Max proportion of spins to flip at any given iteration


        - Returns
            - An array of states (num_chain number of states) at different energies 
        """

        # Initialize state as random state if not provided

        num_chains = len(temperature_list)


        if Unfurled_Current_layer_State is None:
            Unfurled_Current_layer_State = self.Random_state_unfurled()

        # Initialize chains
        chains = [Unfurled_Current_layer_State.copy() for _ in range(num_chains)]
        Previous_layer = [Unfurled_Previous_layer_State.copy() for _ in range(num_chains)]
        # Initialize energies
        energies = np.zeros((num_chains, len(Unfurled_Current_layer_State)))
        
        
        # Run parallel tempering
        for step in range(num_steps):

            ## update all chains-->forward gibbs
            for i in range(num_chains):
                # Perform Gibbs sampling
                temp_for_chain = float(temperature_list[i])
                chains[i] = self.Forward_Gibbs_pass_partitioned(Unfurled_Previous_layer_State, chains[i], partition=partition, temperature=temp_for_chain)
            

            energies = self.Energy_state(Previous_layer,chains)


            # Swap states between chains with different temperatures
            for i in range(num_chains - 1):
                
                ## choose different chain to exchange with
                ## here next temperature step chain is chosen
                j = i+1

                # Calculate swap probability
                # Avoid the overflow issue by hardcoding the power in swap exponential (exp of difference between things converges better)
                Power = (energies[i] - energies[j]) / ((1/temperature_list[i]) - (1/temperature_list[j]))
                if Power > 0:
                    swap_factor = 1
                else:
                    swap_factor = np.exp(Power)

                if np.random.rand() < swap_factor:
                    chains[i], chains[j] = chains[j], chains[i]

        
        return np.array(chains)
    

    
    def Stochastic_Contrastive_Divergence(self, Unfurled_Previous_layer_State_array, Unfurled_Current_layer_State_array, Temperature_list, batch_fraction = 1, num_steps = 20, partition = 0.5, Adaptive = True):
        
        num_chains = len(Temperature_list)
        Num_training_examples = len(Unfurled_Current_layer_State_array)

        #### Check for error in sizes 
        if self.Influence==True and Num_training_examples != len(Unfurled_Previous_layer_State_array):
            print("Mismatch between previous layer and current layer example size. Terminating!")
            return 

        # -------------------------------------------------------------------
        ### If Adaptive==False, then no RMS Prop. Raw learning rate is used
        if Adaptive:
            self.Adaptive = Adaptive
        else:
            ### clear out the stored squared gradients
            self.squared_grad_weights = np.zeros_like(self.weights)
            self.squared_grad_bias = np.zeros_like(self.bias)
            self.squared_influence = np.zeros_like(self.Influence_matrix)


        # Initialize parameter-specific adaptive learning rates if not exist
        if not hasattr(self, 'weight_lr_cache'):
            self.weight_lr_cache = np.ones_like(self.weights) * self.learning_rate
            self.bias_lr_cache = np.ones_like(self.bias) * self.learning_rate
            self.influence_lr_cache  = np.ones_like(self.Influence_matrix) * self.learning_rate

            self.squared_grad_weights = np.zeros_like(self.weights)
            self.squared_grad_bias = np.zeros_like(self.bias)
            self.squared_influence = np.zeros_like(self.Influence_matrix)
            
            self.decay_rate = 0.9  # RMSProp decay rate
            self.epsilon = 1e-6    # Small constant for numerical stability
        #---------------------------------------------------------------------------
        
        ###---------------- Reinforced Hebbian Cycle begins ----------------

        Energy_data = self.Energy_state(Unfurled_Previous_layer_State_array, Unfurled_Current_layer_State_array)
        Mean_energy = np.mean(Energy_data)
        Dev_energy = np.std(Energy_data)
        ### Scale energy by mean
        Scaled_energy = (Energy_data-Mean_energy)/ (Dev_energy + self.epsilon)

        ### emphasize examples at too high energy more
        ### deemphasize examples at too low energy
        P_learn = self.Softmax_probability_stable(Scaled_energy, sign = 1,  temperature=self.temperature)
        P_forget = self.Softmax_probability_stable(Scaled_energy, sign = -1, temperature =self.temperature)

        ## Total probability
        P_eg = (1+ P_learn - P_forget)/ Num_training_examples 

        ### converting P_eg to column vector using np.newaxis to broadcast for multiplication
        ### alternatively I can also reshape this using P_eg.reshape(-1,1) and multiply.

        Scaled_examples = Unfurled_Current_layer_State_array * P_eg[:, np.newaxis]

        ### weights(w), bias(b) and influence(i) update
        dw_hebb = Scaled_examples.T @ Unfurled_Current_layer_State_array
        db_hebb = np.sum(Scaled_examples, axis = 0)
        di_hebb = (Scaled_examples.T@Unfurled_Previous_layer_State_array)

        ##### --------------------- End Hebbian Phase ------------------------------



        #### ------------Anti Hebbian Phase with Parallel Tempering begins ----------
        ### find good starting set- take data states and boil them to different temperatures:
        ### I will parallel temper these states again!
        ### Running Parallel_tempering without specifying a starting states starts it from a random state



        ############# Defining Temperature list for Parallel tempering ######
        #########

        dw_anti_hebb =0
        db_anti_hebb =0
        di_anti_hebb = 0

        ###Pick Training data_subset
        
        Num_pick = int(len(Unfurled_Current_layer_State_array) * batch_fraction) 
        ### error handling --just in case batch fraction is entered as more than 1
        Num_pick = np.min((Num_pick, len(Unfurled_Current_layer_State_array)))


        Picked_indices = np.random.choice(np.arange(0, Num_training_examples), Num_pick, replace = False)
        Current_layer_subsample = Unfurled_Current_layer_State_array[Picked_indices]
        Previous_layer_subsample = Unfurled_Previous_layer_State_array[Picked_indices]

        Energy_subsample = Energy_data[Picked_indices]

        #Collect_chains = []
        #Collect_probability= []
        #Collect_previous = []
        #Min_energy_state = []
        Recons_error = 0


        for i,data in enumerate(tqdm(Current_layer_subsample)):

            Prev_layer_data = Previous_layer_subsample[i]

            Chains = self.Parallel_tempering(Prev_layer_data ,Temperature_list, data, num_steps = num_steps, partition = partition)

            Prev_copy = [Prev_layer_data.copy() for _ in range(num_chains)]
            Current_copy = [data.copy() for _ in range(num_chains)]

            Calculated_energy_of_chain = self.Energy_state(Prev_copy, Chains)

            Probability_chain = self.Softmax_probability_stable(Calculated_energy_of_chain, sign=-1)

            Scaled_chain_data = Chains* Probability_chain.reshape(-1,1)

            Abs_difference = np.sum(np.abs(Current_copy - Chains), axis = 1) * 0.5
            #Scaled_Difference = Abs_difference * Probability_chain.reshape(-1,1)
            #Recons_error += np.sum(Scaled_Difference)/Num_pick

            min_difference = np.min(Abs_difference)
            Recons_error+=min_difference/Num_pick

            ### corrections
            dw_anti_hebb += (Scaled_chain_data.T @ Chains)/Num_pick
            db_anti_hebb += np.sum(Scaled_chain_data, axis = 0)/Num_pick
            di_anti_hebb += (Scaled_chain_data.T @ Prev_copy)/Num_pick

            #min_energy_arg = np.argmin(Calculated_energy_of_chain)

            #Min_energy_state.append(Chains[min_energy_arg])

        ### turning into numpy arrays
        #Min_energy_state = np.array(Min_energy_state)

        ### ------------------------- End Anti Hebbian Phase  ------------------------


        ## ---------- Calculating corrections -----

        Dw_full = (dw_hebb - dw_anti_hebb) *self.fundamental_connectivity
        Dw_full = (Dw_full + Dw_full.T)/2
        Db_full = (db_hebb - db_anti_hebb)
        Di_full = (di_hebb - di_anti_hebb)


        ## Weight Error, Bias error and influence error
    
        W_error = np.sum(np.abs(Dw_full))/ np.sum(self.fundamental_connectivity)
        B_error = np.mean(np.abs(Db_full))
        I_error = np.mean(np.abs(Di_full))



        #### check if nan:

        if math.isnan(W_error) or math.isnan(B_error) or math.isnan(I_error)  :
            print("NaN encountered!! No Update")
            Dw_full = 0
            Db_full = 0
            Di_full = 0

        ### -----------Adaptive learning rates ------------
        
        if self.Adaptive==True:
            
            # RMSProp-like update for adaptive learning rates
            self.squared_grad_weights = self.decay_rate * self.squared_grad_weights + (1 - self.decay_rate) * (Dw_full ** 2)
            self.squared_grad_bias = self.decay_rate * self.squared_grad_bias + (1 - self.decay_rate) * (Db_full ** 2)

            self.squared_influence = self.decay_rate * self.squared_influence + (1 - self.decay_rate) * (Di_full ** 2)
            
            # Compute adaptive learning rates
            adaptive_lr_weights = self.learning_rate / (np.sqrt(self.squared_grad_weights) + self.epsilon)
            adaptive_lr_bias = self.learning_rate / (np.sqrt(self.squared_grad_bias) + self.epsilon)
            adaptive_lr_influence = self.learning_rate /(np.sqrt(self.squared_influence) + self.epsilon)

            self.weight_lr_cache = adaptive_lr_weights
            self.bias_lr_cache = adaptive_lr_bias
            self.influence_lr_cache = adaptive_lr_influence

            
        else:
            adaptive_lr_weights = self.learning_rate
            adaptive_lr_bias = self.learning_rate
            adaptive_lr_influence = self.learning_rate


        ### with these learning rates, we now modify the weights and biases
        self.weights += adaptive_lr_weights * Dw_full
        self.bias += adaptive_lr_bias * Db_full
        
        self.Influence_matrix += adaptive_lr_influence * Di_full 


        ## reconstruction error
        ## although this is not that helpful
        #Recons_error = np.mean((np.sum(np.abs(Current_layer_subsample - Min_energy_state), axis = 1))/2)



        return W_error, B_error, I_error, Recons_error
