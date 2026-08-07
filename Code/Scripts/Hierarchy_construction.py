
import numpy as np
import matplotlib.pyplot as plt
from jenkspy import JenksNaturalBreaks
from tqdm import tqdm

## Uses VNE bootstrap to partition class
## Also contains Kernel-Density estimate class for feature partition.

class Hierarchy_Analysis():
    def __init__(self, Compact_Data, Number_of_clusters=5, Min_partition_size = 2):
        ## import packages
        self.Compact_data = Compact_Data
        self.Number_of_cluster = Number_of_clusters
        self.alphabets = np.unique(Compact_Data)
        self.minimum_size_of_partition = Min_partition_size

        if Min_partition_size>1:
            print(f"Any Entropic Scale with less than {Min_partition_size} features will be combined with the nearest scale.")

        self.Number_of_letters = len(self.alphabets)
        self.Data_size , self.Number_of_features = np.shape(Compact_Data)
        self.Total_dof = int(self.Number_of_letters * self.Number_of_features)
        # Data Unfurling
        self.Unfurled_Data = self.Unfurl_Data(self.Compact_data)
        # Computing the Shannon Entropy
        self.Entropy = self.Shannon_Entropy(self.Unfurled_Data, self.Number_of_letters)
        # Jenks Fisher Partition of the data
        self.Labels, self.Brakes, self.Goodness_of_variance, self.Partition_map_raw  = self.Jenks_Fisher_Partition(self.Entropy, self.Number_of_cluster)
        self.Partition_map = self.Partition_Map_Cleaner(self.Partition_map_raw)
        self.Number_of_cluster = len(self.Partition_map)
        if len(self.Partition_map)!= len(self.Partition_map_raw):
            print("Clean up was necessary. Check Raw partition map vs Partition map")
            print("Number of clusters =", self.Number_of_cluster)

    def Return_Data_Properties(self, verbose = True):
        """--Returns: 
        - Unfurled Data, Partition_map of features and Alphabet list
        - if verbose==True, also prints other data properties
        --------------------------------------------------"""

        if verbose:
            print("--- Data Properties----")
            print("Number of Alphabets  =", self.Number_of_letters)
            print("Number of input Partitions =", self.Number_of_cluster)
            print("Goodness of variance (Jenks Fisher)=", self.Goodness_of_variance)

        return self.Unfurled_Data, self.Partition_map , self.alphabets
    def Return_Entropic_Properties(self, verbose = True):
        """--Returns: 
        - Shannon Entropy per feature for the entire data
        - Jenks Fisher -Brakes
        - Jenks Fisher - Labels
        --------------------------------------------------"""
        if verbose:
            print("Number of clusters (Jenks Fisher) =",  self.Number_of_cluster)
            print("Goodness of variance (Jenks Fisher)=", self.Goodness_of_variance)
        return self.Entropy, self.Brakes, self.Labels

    def Entropy_Bootstrap(self,  Data_fraction=0.5,  Number_of_Trials = 500, Replace=True):
        """Rather than relying on the Jenks Fisher Algorithm,
          we will perform VNE bootstrap on the subsample of the data
          -- Input: 
            - Data_fraction = Fraction of Total data to subsample (between 0 and 1)
            - Number of Trials= Total number of subsample + VNE calculation Trials
            - window = Standard deviation window (1 - 2) to consider for connection
            - Replace =True/False . if True Bootstrap, if False, subsample
            """
        Total_Data_set = self.Unfurled_Data
        Total_Data_size = self.Data_size
        ## --------------------   idiot-proofing. ----------------
        Data_fraction = np.min((0.9, np.abs(Data_fraction))) # At most pick 90% of data
        Number_to_choose = int(Total_Data_size * Data_fraction)+ 1 #<- Plus 1 so that number chosen is not 0  
        Number_of_Trials = int(Number_of_Trials)
        VNE_Array = []

        for i in tqdm(range(0, Number_of_Trials)):
            # Choose data
            Data_arg_chosen = np.ravel(np.random.choice(np.arange(0, Total_Data_size), Number_to_choose, replace = Replace))
            Data_chosen = Total_Data_set[Data_arg_chosen]
            ## compute VNE
            VNE_chosen = self.Shannon_Entropy(Data_chosen, self.Number_of_letters)
            VNE_Array.append(VNE_chosen)

        VNE_Array = np.array(VNE_Array)
        Mean_VNE = np.mean(VNE_Array, axis = 0)
        Deviation_VNE = np.std(VNE_Array, axis=0)

        self.Mean_VNE = Mean_VNE
        self.St_dev_VNE = Deviation_VNE

        return np.array(VNE_Array), Mean_VNE, Deviation_VNE

    def find_elbow(self, X_Array, Y_Array):
        # Vector between first and last point
        line_vec = np.array([X_Array[-1] - X_Array[0], Y_Array[-1] - Y_Array[0]])
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        distances = []
        for i in range(len(X_Array)):
            # Vector from first point to current point
            point_vec = np.array([X_Array[i] - X_Array[0], Y_Array[i] - Y_Array[0]])
            # Project point_vec onto line_vec
            projection = np.dot(point_vec, line_vec_norm) * line_vec_norm
            # Find orthogonal distance component
            ortho_vec = point_vec - projection
            distances.append(np.sqrt(np.sum(ortho_vec**2)))
        # The elbow is the point with the maximum distance
        Elbow_value = X_Array[np.argmax(distances)]
        return Elbow_value
    
    def Reset_Jenks(self, Number_of_clusters):
        """This is so that we can reset the internal number of clusters and recompute jenks partitions
        -- Input : New number of clusters (int)
        --------------------------------------------------"""
        ##This is to change number of clusters
        self.Number_of_cluster = Number_of_clusters
        self.Labels, self.Brakes, self.Goodness_of_variance, self.Partition_map_raw  = self.Jenks_Fisher_Partition(self.Entropy, self.Number_of_cluster)
        self.Partition_map = self.Partition_Map_Cleaner(self.Partition_map_raw)
        self.Number_of_cluster = len(self.Partition_map)
        if len(self.Partition_map)!= len(self.Partition_map_raw):
            print("Clean up was necessary. Check Raw partition map vs Partition map")
            print("Number of clusters =", self.Number_of_cluster)
    

    def Overlap_Test(self, low, GVF_array, sigma_cutoff = 1):
        Mean_GVF = np.mean(GVF_array, axis = 1)
        Dev_GVF = np.std(GVF_array, axis =1)

        ###  Lets first look at Elbow test:
        X_range = np.arange(low, low+len(Mean_GVF))
        Y_vals = Mean_GVF
        Appropriate_cluster_number_elbow= self.find_elbow(X_range, Y_vals)
        

        ### ------------
        ## Now lets look at histogram
        Overlap = False
        for num in range(0,len(GVF_array)-1):
            Appropriate_cluster_number_overlap = num+low
            
            mu_self = Mean_GVF[num]
            dev_self = Dev_GVF[num]
            mu_next = Mean_GVF[num +1]
            dev_next = Dev_GVF[num +1]
            Condition = ((mu_self + sigma_cutoff*dev_self) >= (mu_next - sigma_cutoff*dev_next))

            if Condition == True:
                Overlap = True
                print("Overlap condition met.")
                ## use the mean of the two methods rounded up
                Appropriate_cluster_number = np.ceil(np.mean([Appropriate_cluster_number_elbow,Appropriate_cluster_number_overlap]))
                break
        print("Appropriate number of clusters from Elbow.  =", Appropriate_cluster_number_elbow)
        print("Appropriate number of clusters from Overlap =", Appropriate_cluster_number_overlap)

        if Overlap == False:
            if Appropriate_cluster_number_elbow < X_range[-1]:
                # elbow test passed but overlap didn't.
                # compensate by adding 1 to elbow test
                print("Elbow test Passed, overlap test didn't.")
                print("Compensating elbow scale by 1")
                Appropriate_cluster_number =  Appropriate_cluster_number_elbow +1
            else:
                print("Elbow condition not met")
                print("Overlap condition not met.")
                print("Number of cluster set to maximal value tested so far")
                print("Both Elbow and Overlap test failed. Increase largest number of scales.")

        
        return Appropriate_cluster_number


        
    def Jenks_GVF_Bootstrap(self, Lowest_Number_of_scales, Highest_Number_of_scales, Data_fraction=0.5,  Number_of_Trials = 100, sigma_cutoff = 1):
        """ Given:
            Lowest_Number_of_scales(int): Smallest Number of clusters used in bootstrap
            Highest_Number_of_scales(int): Largest Number of clusters used in bootstrap
            Data_fraction (float) : (between 0 and 1), the random fraction of data used for bootstrap (defaule = 0.5)
            Number_of_Trials(int): Total number of bootstrap trials for the above parameters (defaule = 100)
            sigma_cutoff (float): Min(overlap between the histograms) needed to select appropriate # of scale (default =1) 
    
        This outputs - a dictionary with  Goodness of variance fit for Jenks fisher for each trial 
                     - Appropriate number of scales.
                     - Feature Partition map (with singleton clean up) with these many number of scales
        --------------------------------------------------"""
        Total_Data_set = self.Unfurled_Data
        Total_Data_size = self.Data_size
        ## --------------------   idiot-proofing. ----------------
        Data_fraction = np.min((0.9, np.abs(Data_fraction))) # At most pick 90% of data
        Number_to_choose = int(Total_Data_size * Data_fraction)+ 1 #<- Plus 1 so that number chosen is not 0  
        Number_of_Trials = int(Number_of_Trials)
    
        Lowest_Number_of_clusters = int(Lowest_Number_of_scales)
        Highest_Number_of_clusters = int(Highest_Number_of_scales)

        Num_small, Num_large = np.sort((Lowest_Number_of_clusters, Highest_Number_of_clusters))

        if Num_small == Num_large:
            print("Smallest and Largest number of scale cannot be the same. Change values")
            return

        ## --------------------------------------------------------
        Clusters = np.arange(Num_small, Num_large+1,1)
        
        num_clust = len(Clusters)

        GVF_array = np.zeros(shape = (num_clust, Number_of_Trials))
        print(np.shape(GVF_array))


        for i in tqdm(range(0, Number_of_Trials)):
            # Choose data
            Data_arg_chosen = np.ravel(np.random.choice(np.arange(0, Total_Data_size), Number_to_choose, replace = False))
            Data_chosen = Total_Data_set[Data_arg_chosen]
            ## compute VNE
           
            Entropy_chosen = self.Shannon_Entropy(Data_chosen, self.Number_of_letters)
            
            Cl = []
            for c in Clusters:
                _,_, gvf, _ =  self.Jenks_Fisher_Partition(Entropy_chosen, c)
                Cl.append(gvf)

            GVF_array[:,i]=Cl
        ##--------- major bootstrap done -----------    
        ## It is clearer if we provided a dictionary rather than the array
        GVF_Array_dictionary={}
        for i,c in enumerate(Clusters):
            name = str(c)+"_Entropic_Scales"
            GVF_Array_dictionary[name] = GVF_array[i]
        Appropriate_cluster_number = self.Overlap_Test(Lowest_Number_of_clusters, GVF_array, sigma_cutoff)
        print("Returning- dictionary containing GVF Bootstraps for various cluster numbers and appropriate cluster numbers")

        ## Lets actually compute the partition map as well:
        Labels, Brakes, Goodness_of_variance, Partition_map_raw  = self.Jenks_Fisher_Partition(self.Entropy, Appropriate_cluster_number)
        Clean_Partition_map = self.Partition_Map_Cleaner(Partition_map_raw)

        return GVF_Array_dictionary, Appropriate_cluster_number, Clean_Partition_map

    ##------------- Internal Functions that seldom needs to be called ------
    ### -------- Unfurling the data-------------
    def make_unfurled_state(self, State):
        """----- Compact State unfurler ----
        --- takes compact state from MSA in alphabet notation
        --- takes alphabet list in order (alphabet list size = n)
        --- takes subsequent alphabet into one hot encoded basis
        """

        Ordered_alphabet_list = self.alphabets
        num_alphabet = self.Number_of_letters
        length_unfurled = self.Total_dof
        unfurled_state = []

        for value in State:

            alphabet_unfurled = np.zeros(num_alphabet)

            arg_where_one = np.argwhere(Ordered_alphabet_list ==value)
            alphabet_unfurled[arg_where_one] =1
            unfurled_state.append(alphabet_unfurled)

        unfurled_state = np.ravel(np.array(unfurled_state))

        return unfurled_state

    def Unfurl_Data(self, Compact_data):
        """-- Given the compact dataset (MSA)
        -This unfurles the entire dataset. (Constructs one-hot-encoded data)
        - Repeated calls of single sequence unfurler function (make_unfurled_state)
        """
        Unfurled_data=[]
        for state in Compact_data:
            state_unf = self.make_unfurled_state(state)
            Unfurled_data.append(state_unf)
        return np.array(Unfurled_data)

    ### -------------------------
    #### ------------- Entropy Calculator, Jenks Fisher and Partition creation ----

    def Shannon_Entropy(self, Unfurled_data, Num_Letters):
        """Given 
        - Unfurled Data : Entire Data array written in on-hot-encoded format
        - Num_Letters (int): Number of possible letters per position (Internal degree of freedom)
        - This computes the Shannon Entropy per site for the data."""

        Data_size, Total_dof = np.shape(Unfurled_data)
        Entropy_array = []
        for i in range(0, Total_dof, Num_Letters):
            start = i
            end = i + Num_Letters
            Part_i = Unfurled_data[:, start:end]

            ## compute eigenvalues
            _,sigma_i, _ = np.linalg.svd(Part_i)
            
            normed_lambda = (sigma_i**2 / Data_size)
            ## epsilon for stabilization of log
            eps = np.max([np.min(normed_lambda)/100, 1e-8])
            vne_i = -1*np.dot(normed_lambda, np.log2(normed_lambda + eps))
            Entropy_array.append(vne_i)

        return np.array(Entropy_array)


    def Partition_Map_Cleaner(self, Partition_Map):
        """ Given:
        1. Partition Map= Optimal Jenks Fisher Partition map
        2.  Min_size = minimal admissible number of features in a partition
        - This collapses any partition with size less than Min_size by adding it to the next higher entropy partition 
        - If the partition to collapse happens to be the last one, it gets added to the previous high entropy partition
        - Choice of Min_size = 1 should leave partition unchanged"""
        Map = Partition_Map
        Min_partition_size = self.minimum_size_of_partition
        ### 1) Find number of elements in each partition:
        ###
        Partition_Map_copy = Map.copy()
        New_Partition_Map = []
        for i in range(0,len(Partition_Map)):
            s_i = np.sum(Partition_Map_copy[i])
            if s_i < Min_partition_size and i < len(Map)-1:
                ## merge with next
                p_i = Map[i]
                p_i_next = Map[i+1]
                Partition_Map_copy[i+1] += p_i
                Partition_Map_copy[i] *=0

            else:
                New_Partition_Map.append(Partition_Map_copy[i])
            
        ### Check to see if the last feature is a singleton set
        Sum_check = np.sum(New_Partition_Map, axis =1)

        if Sum_check[-1] < Min_partition_size:
            New_Partition_Map[-2]+=New_Partition_Map[-1]
            New_Partition_Map = np.array(New_Partition_Map[0:-1])
        else:
            New_Partition_Map = np.array(New_Partition_Map)
        return New_Partition_Map
    

        
    def Jenks_Fisher_Partition(self, Shannon_Entropy, Num_Clusters):
        """ Computes Jenks Fisher Partition of the given effective dimension array with given number of clusters:

            Inputs:
            - Shannon_Entropy : Array containing Shannon Entropy per site
            - Num_Clusters (int) : Number of clusteres we want to partition the effective dimension array into

            - Effective dimension is calculated internally from Shannon Entropy
        """
        
        Num_Clusters = int(Num_Clusters)
        Number_of_features = len(Shannon_Entropy)
        Effective_Dimension = 2**Shannon_Entropy
        jnb = JenksNaturalBreaks(Num_Clusters) 
        jnb.fit(Effective_Dimension)
        GVF = jnb.goodness_of_variance_fit(Effective_Dimension)  #<- Goodness of Variance Fit for Jenks Fisher Algorithm
        Labels = jnb.labels_
        Brakes = jnb.breaks_
        ###
        Labels_unique = np.sort(np.unique(Labels))
        Partition_map = []
        for i,l in enumerate(Labels_unique):
            part_i = np.zeros(Number_of_features)
            Args = np.argwhere(Labels ==l)
            part_i[Args] = 1
            Partition_map.append(part_i)

        Partition_map = np.array(Partition_map)

        return Labels, Brakes , GVF, Partition_map
    # -----------------------------------------------------------------------
## What if we used Kernel Density Estimates to do feature partitioning?
## This is most useful for sentences and MNIST
## This needs Mean value array and Standard Deviation array of the Entropy from Hierarchy module


#### Now lets do KDE
from scipy.signal import find_peaks
from scipy.stats import norm

class KDE_Peak_Finder():
    def __init__(self,Mean_values, Deviations, X_Range, peak_height_cutoff = 0.01):


        self.mean = Mean_values
        self.num_features = len(self.mean)
        self.deviation = Deviations
        self.x_range = X_Range
        self.num_step = len(X_Range)
        smallest_mean = min(self.mean)
        largest_mean = max(self.mean)
        largest_dev = max(self.deviation)

        ## the of values over which the distribution sits. 
        if (smallest_mean-largest_dev)<np.min(X_Range) or (largest_mean+largest_dev)>np.max(X_Range) :
            print("Range given is not sufficient. Redefining")
            low_val = (smallest_mean -3*largest_dev)
            high_val = (largest_mean + 3*largest_dev)
            self.x_range = np.linspace(low_val, high_val,self.num_step)

        self.delta_x = abs(self.x_range[1]- self.x_range[0])
        self.data = zip(self.mean, self.deviation)
        self.num_data = len(self.mean)

        self.density = self.Density_Construction()
        self.peak_height_cutoff =  peak_height_cutoff 
        self.cluster_center, self.cluster_property = self.Peak_Finder()
        self.num_peaks = len(self.cluster_center)
        self.cluster_height = self.cluster_property['peak_heights']
        self.peak_assignment, self.assignment_dictionary = self.Peak_Assignment()
        self.Partition_Map = self.Partition_Map_Creator()


    def Density_Construction(self):
        """
        This function:
        -puts a gaussian distribution at each mean value.
        - the deviation of each gaussian is equal to the corresponding deviation in deviation array
        - The gaussian is defined over the x_range
        - In the end, it sums up all of these gaussians
        - Then it normalizes the distribution to give the mixture model
        """
        # Initializing the density array over the x range
        density = np.zeros_like(self.x_range)
        # Adding a Gaussian for each point: N(mu = mean_i, sigma = dev_i)
        for mu, sigma in self.data:
            density += norm.pdf(self.x_range, loc=mu, scale=sigma)
        #(Normalization)
        return density / self.num_data

    def Peak_Finder(self):
        """This function:
        - Finds the peaks of the mixture model"""
        peak_index, cluster_property = find_peaks(self.density, height = self.peak_height_cutoff)
        cluster_center = self.x_range[peak_index]
        return cluster_center, cluster_property

    def Peak_Assignment(self):
        """ This function:
            - assigns each data point (mean) to the nearest detected peak"""
        assignments = []
        for point in self.mean:
            # Find the distance from this point to all detected peaks
            distances = np.abs(point - self.cluster_center)
            closest_peak_idx = np.argmin(distances)
            assignments.append(closest_peak_idx)
        assignments = np.array(assignments)
        
        assignment_dict={}
        for i in range(0, self.num_peaks):
            Arg_i = np.argwhere(assignments==i).reshape(-1)
            assignment_dict[i] = Arg_i
        return assignments, assignment_dict
    
    def Partition_Map_Creator(self):
        """Creates a entropy partition map 
        - the rows are each scale of entropy
        - the columns are features in the data
        - 0 => feature not included in that scale
        -1 => feature included in that scale"""
        Assignment_dictionary = self.assignment_dictionary

        #total_features = len(np.concatenate(list(Assignment_dictionary.values())))
        total_features = self.num_features
        total_scales = len(Assignment_dictionary.keys())
        Partition_Map = np.zeros((total_scales, total_features))

        for i,k in enumerate(Assignment_dictionary.keys()): 
            vals_k = Assignment_dictionary[k]
            Partition_Map[i][vals_k] =1
        return Partition_Map

    def Partition_Plotter(self):
        """This function:
            - plots the Entropic Scales
            - Shows which features is included/not included in which scale """
        plt.figure(figsize=(10,5))
        img = plt.imshow(self.Partition_Map, cmap = 'binary', aspect = "auto")
        plt.title(f"{len(self.Partition_Map)} Scales of Entropy From KDE Model", fontsize = 15)
        plt.xlabel("Features", fontsize = 15)
        plt.ylabel("Scale", fontsize = 12)
        plt.yticks(range(0, len(self.Partition_Map)), range(1, len(self.Partition_Map)+1))
        cbar = plt.colorbar(img, boundaries=[-0.5, 0.5, 1.5], ticks=[0, 1], fraction = 0.01)
        cbar.set_ticklabels(['Not included', '   Included'])
        plt.show()
    
    def Plotter(self, show_individual = True):
        """This function:
            - plots the total pdf 
            - if show_individual==True, plots individual pdfs
            - Plots the peak locations """
        fig,ax = plt.subplots(figsize=(6,3))
        ax.plot(self.x_range,self.density, color = "black", linewidth = 2.5, zorder = 1, label = "Total PDF")
        ax.scatter (self.cluster_center, self.cluster_height, color = "darkorange", edgecolor = "black",marker = "o", zorder = 2 , label = "Cluster Centers")

        #-----------------
        if show_individual:
            for i in range(0, self.num_data-1):
                if i == 0:
                    label_i = "Individual PDF"
                else:
                    label_i = None
                density_i = norm.pdf(self.x_range, loc=self.mean[i], scale=self.deviation[i]) * (1/self.num_data)
                ax.plot(self.x_range,density_i, color = "lightcoral", zorder =0, linewidth = 0.4, label =label_i)
        #----------------
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("Quantity")

        ax.set_title(f"Mixture Model with {self.num_peaks} peaks")
        ax.legend()
        ax.grid(alpha = 0.3, zorder = 0)
        plt.show()

    def Results(self, show_individual= True):
        print("Number of Input Pdfs=", self.num_data)
        print("Number of Peaks Found=", self.num_peaks)

        for i in range(0, self.num_peaks):
            print(f"Peak {i} --> {len(self.assignment_dictionary[i])} Point(s) Assigned:" ,self.assignment_dictionary[i])    
        self.Plotter(show_individual)
        