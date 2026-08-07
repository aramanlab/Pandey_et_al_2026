import numpy as np
import matplotlib.pyplot as plt
from jenkspy import JenksNaturalBreaks
from tqdm import tqdm
from scipy.stats import poisson


### this also contains the death of individuals with 0 fitness at every epoch.


##### Jenks Fisher for partition ####
def Jenks_Fisher_Partition(Shannon_Entropy, Num_Clusters):
    Effective_Dimension = 2**Shannon_Entropy

    jnb = JenksNaturalBreaks(Num_Clusters) 

    jnb.fit(Effective_Dimension)

    Labels = jnb.labels_
    Brakes = jnb.breaks_

    return Labels, Brakes

#################################



class Evolution_simulation:

    ### Environment dependent functions

    def __init__(self, Feature_Partition, Number_of_Letters, selection_strength=1):

        """Feature Partition- Numpy array that tells how many features are in each splitting of the constraint tree
        Say [1,3,2]- then first split = 1=> single position difference 
                        second split = 3=> three positions difference
                         third split = 2 => 2 positions different
        - Then total number of features = 1+3+2= 6 loci
        - Number of letters: int = Total number of letters that can go into each site """


        self.Feature_Partition = np.array(Feature_Partition)//1 
        self.selection_strength = selection_strength
        self.num_sites = int(np.sum(self.Feature_Partition))
        self.layers = len(self.Feature_Partition)
        self.cumulative_feature_partition = np.cumsum(np.concatenate(([0], self.Feature_Partition)))

        self.num_letters = Number_of_Letters
        self.Letters = np.arange(0, self.num_letters)
        self.Constraints = self.Generate_Constraints()
        self.Fitness_Sequence = self.Constraint_Sequence_Generator()

        

    def Generate_Constraints(self):

        """This generates a constraint sequence for the features
        - Bifurcation map
        - Or a binary Tree"""

        Letters = self.Letters
        Features_Partition = self.Feature_Partition 
        Layers = self.layers
        Full_constraints=[]
        Proportions=[]
        for l in range(0, Layers):
            num_features = Features_Partition[l]
            Constraints = []
            #num_constraints = 2**(l+1)  #< - this starts from 2 contraints at root
            num_constraints = 2**(l)    #< - thi strats with 1 constraint at root


            for n in range(0,num_constraints):
                d1 = np.random.choice(Letters, num_features)
                Constraints.append(d1)
            Full_constraints.append(Constraints)
        self.Constraints = Full_constraints
        return Full_constraints



    def Constraint_Sequence_Generator(self):
        """Converts Contraint tree into constraint sequence.
          - This is easier to visualize as sequences
          - This is also easier to operate on when Hamming for fitness"""

        Constraints = self.Constraints

        # All the high fitness sequences go into this array
        Fit_seq = np.zeros(shape=(2**self.layers, self.num_sites))

        cumulative_features = self.cumulative_feature_partition

        for layer in range(0, self.layers):
            const = Constraints[layer]
            start_col = cumulative_features[layer]
            end_col = cumulative_features[layer+1]

            for i in range(0, len(const)):
                c = const[i]

                ## in how many steps does this constraint layer divide the fitness sequences?
                steps  = len(Fit_seq)//len(const) 
                # we have to walk in that many steps for each constraint sequence
                start_row = i* steps
                end_row =  (i+1)*steps

                Row= Fit_seq[start_row:end_row]

                Row[:, start_col:end_col] = c
                Fit_seq[start_row:end_row]=Row

        return Fit_seq
    
    def Constraint_Sequence_Generator_1(self, Constraints):
        """Converts Contraint tree into constraint sequence.
          - This is easier to visualize as sequences
          - This is also easier to operate on when Hamming for fitness"""


        # All the high fitness sequences go into this array
        Fit_seq = np.zeros(shape=(2**self.layers, self.num_sites))

        cumulative_features = self.cumulative_feature_partition

        for layer in range(0, self.layers):
            const = Constraints[layer]
            start_col = cumulative_features[layer]
            end_col = cumulative_features[layer+1]

            for i in range(0, len(const)):
                c = const[i]

                ## in how many steps does this constraint layer divide the fitness sequences?
                steps  = len(Fit_seq)//len(const) 
                # we have to walk in that many steps for each constraint sequence
                start_row = i* steps
                end_row =  (i+1)*steps

                Row= Fit_seq[start_row:end_row]

                Row[:, start_col:end_col] = c
                Fit_seq[start_row:end_row]=Row

        return Fit_seq


    def Weight_assignment(self, Weight_partition, noise_factor = 0.05):

        """Assigns noisy weight to each feature
        -- this creates variable fitness contribution
        """
        Features_partition = self.Feature_Partition 

        self.Weight_partition = Weight_partition[0:len(Features_partition)]

        Weight = np.zeros(np.sum(Features_partition))
        cumulative_features = self.cumulative_feature_partition

        ## making sure noise factor is positive
        noise_factor = np.abs(noise_factor)
        noise_factor = np.min((0.99, noise_factor))


        for layer in range(0, len(Features_partition)):
            weight_center = Weight_partition[layer]
            num_features = Features_partition[layer]

            noise_level = np.random.uniform(-noise_factor* weight_center, noise_factor*weight_center, num_features)

            start_col = cumulative_features[layer]
            end_col = cumulative_features[layer+1]

            Weight[start_col: end_col] = weight_center + noise_level
    
        self.Weights = Weight

        return Weight

        ###
    def Fitness_Measurement_0_old(self, sequence):
        Weights = self. Weights
        Fitness_Sequences = self.Fitness_Sequence
        """Here we want to measure the fitness of the sequence 
        based on the weights and hamming distance to Fitness_sequences"""

        Similarity=[]
        for seq in Fitness_Sequences:

            sim = 1*(sequence==seq) 

            Similarity.append(sim)
        Similarity = np.array(Similarity)
        Fitness_array = Similarity@ Weights

        ##Fitness is the maximum possible value among these:
        Fitness_max = float(np.max(Fitness_array))
        #print(Fitness_array)
        return Fitness_max
    
    def Fitness_Measurement_0(self, sequence):
        """Weighted Hamming distance - OPTIMIZED"""
        # Vectorized comparison across all fitness sequences at once
        similarities = (self.Fitness_Sequence == sequence).astype(float)  # Shape: (n_seqs, seq_len)
        fitness_array = similarities @ self.Weights  # Matrix multiplication
        return float(np.max(fitness_array))
    
    

    def Fitness_Measurement_Corrected(self, sequence):
        """Corrected conditional hamming fitness
        Here we want to measure the fitness of the sequence 
        based on the weights and conditional hamming distance to Fitness_sequences
        - Say F0 is the weighted hamming to the highest fitness three positions
        - Say F1 is the weighted hamming to the second highest fitness three positions
        - and so on (for 12 positions divided into 4 groups highest to lowest contributors)
        => Fitness = F0.(1+ F1.(1+ F2.(1+ F3)))
        => Fitness = F0 + F0 F1 + F0 F1 F2 + F0 F1 F2 F3 """

        Weights = self.Weights
        Fitness_Sequences = self.Fitness_Sequence
        Partition_cumulative = self.cumulative_feature_partition
        
        Similarity = []
        for seq in Fitness_Sequences:
            sim = 1 * (sequence == seq)
            Fit_val = sim * Weights
            
            # Calculate layer scores
            layer_scores = []
            for i in range(len(Partition_cumulative) - 1):
                F_i = np.sum(Fit_val[Partition_cumulative[i]:Partition_cumulative[i+1]])
                layer_scores.append(F_i)
            
            # Compute hierarchical fitness: F0 + F0*F1 + F0*F1*F2 + ...
            total_fitness = 0.0
            cumulative_product = 1.0
            
            for F_i in layer_scores:
                cumulative_product *= F_i
                total_fitness += cumulative_product
            
            Similarity.append(total_fitness)
        
        Fitness_array = np.array(Similarity)
        return float(np.max(Fitness_array))


    def Fitness_Measurement_fast(self, sequence):
        """Fully optimized - vectorized where possible"""
        # Vectorize the similarity calculation
        similarities = (self.Fitness_Sequence == sequence).astype(float)
        weighted_similarities = similarities * self.Weights
        
        max_fitness = 0.0
        
        for fit_val in weighted_similarities:
            # Calculating all layer scores at once
            layer_scores = np.array([
                np.sum(fit_val[self.cumulative_feature_partition[i]:
                            self.cumulative_feature_partition[i+1]])
                for i in range(len(self.cumulative_feature_partition) - 1)])
            
            # Using numpy's cumprod for cumulative products
            cumulative_products = np.cumprod(layer_scores)
            total_fitness = np.sum(cumulative_products)
            
            max_fitness = max(max_fitness, total_fitness)
        
        return float(max_fitness)
    
    def Fitness_Measurement(self, sequence):
        """Vectorized hierarchical fitness - instance method"""
        # Vectorize similarity
        similarities = (self.Fitness_Sequence == sequence).astype(float)
        weighted_similarities = similarities * self.Weights
        
        # Calculate layer scores
        n_layers = len(self.cumulative_feature_partition) - 1
        Fitness_layer_score = np.column_stack([
            np.sum(weighted_similarities[:, self.cumulative_feature_partition[i]:
                                          self.cumulative_feature_partition[i+1]], axis=1)
            for i in range(n_layers)
        ])
        
        # Hierarchical fitness
        cumulative_products = np.cumprod(Fitness_layer_score, axis=1)
        total_fitness = np.sum(cumulative_products, axis=1)
        
        return float(np.max(total_fitness))


    ## Population dependent functions:
    #### ---- Mutations ---
    def Set_mutation_rate(self, Mutation_rate, Mutation_type = 0):
        """Setting Mutation Rate and Mutation type
            ---Mutation rate = float 0< Mutation_rate<1 
            ---Mutation type = 0 or 1
                if Mutation type ==0 Standard mutation(default)
                    - each site is independently mutated based on uniform distribution<=mutation rate
                if Mutation type ==1 Poisson Type
                    - Mutation rate=> Average fraction of sites mutating in every reproduction
                    - These site are selected at random. Once selected they are definitely mutated
                 """

        if Mutation_rate>1:
            print("Mutation rate is more than 1. Setting it to max value of 1")

        Mutation_rate = np.min((1, Mutation_rate))

        Mutation_type = Mutation_type//1

        if Mutation_type<1:
            Mutation_type = 0
        else:
            Mutation_type =1

        self.mutation_rate = Mutation_rate
        self.mutation_type = Mutation_type


    def mutate_sequence(self, sequence):
        """Apply mutations based on 
            - independent site assumption
            - constant and same mutation rate for all sites
            - No site is special 
            - Here mutation rate = what is the chance a given site can get mutated"""
        
        letters = self.Letters
        mutation_rate = self.mutation_rate

        mutated = sequence.copy()

        for i in range(0, len(sequence)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.choice(letters)

        return mutated


    def mutate_sequence_Poisson(self, sequence):
        """Apply mutations based for N-sites picked at random
        - a) The probability for all sites to mutate at the same time should be very low 0
        - b) So we will pick the number of sites from a Poisson Distribtuion with parameter (lambda= L* mutation_rate)
             -  P(num_sites_for_possible_mutation) = Pick from Poisson Distribution for any sequence
        - c) Pick these many number of sites at random for this sequence
        - d) For each of these sites, mutate based on uniform probability """

        letters = self.Letters
        mutation_rate = self.mutation_rate
        seq_length = len(sequence)
        mutated = sequence.copy()
        # we have a different interpretation of mutation rate here
        # mutation rate=  what percent of sites on average mutate each time reproduction happens
        # so lambda_parameter = Total number of site * mutation_rate
        lambda_parameter = seq_length * mutation_rate
        # say mutation phenomena = sites gets kicked by some radiation...
        ## then we draw from the poisson distribution --> how many sites get mutated due to this kick
        Choices = np.arange(0, seq_length)
        Prob_mass = poisson.pmf(k= Choices, mu=lambda_parameter)
        Prob_mass= Prob_mass/np.sum(Prob_mass)
        # This is the number of sites out to total length where some 'mutation phenomena' happens
        num_sites_to_mutate = np.random.choice(Choices, p=Prob_mass)

        # These are the sites that get mutated.
        Sites_to_mutate  = np.random.choice(Choices, num_sites_to_mutate, replace = False)

        for site in Sites_to_mutate:
            mutated[site] = np.random.choice(letters)

        return mutated

    # ---------------- End Mutation Functions ------------

    # ---------------- Definining Population  ------------

    def Population(self, Population_size):
        Population_size = int(Population_size)
        if Population_size<1:
            print("Check Population size!")
            return

        Letters = self.Letters
        Num_Features = self.num_sites

        Population = np.random.choice(Letters, size = (Population_size, Num_Features))
        return Population


    # ---------------- Setting up Tournament -------------

    def Set_compete_fraction(self, Competition_fraction):

        if Competition_fraction>1:
            print("Competition rate is more than 1. Setting it to max value of 1")
            Competition_fraction = np.min((1, Competition_fraction))

        ### The fraction of population that competes
        Competition_fraction = np.min((1, Competition_fraction))  #=> competing population isn't larger than total population
        Competition_fraction = np.max((0, Competition_fraction)) #=> competing population isn't smaller than 0 

        self.Competition_fraction = Competition_fraction


    #### -----------------
    def Tournament_single_round_probabilistic(self, Population):
        """
        Fully vectorized tournament with probabilistic duels
        """
        fraction_competing = self.Competition_fraction
        Mutation_type = self.mutation_type
        Total_population = len(Population)
        Competing_population_size = int(fraction_competing * Total_population)
        
        # Ensure even number
        if Competing_population_size % 2 == 1:
            Competing_population_size += 1
        Competing_population_size = min(Competing_population_size, Total_population)
        if Competing_population_size % 2 == 1:
            Competing_population_size -= 1
        
        if Competing_population_size < 2:
            return Population
        
        # Select fighters
        Fighters_arg = np.random.choice(Total_population, size=Competing_population_size, replace=False)
        Fighters = Population[Fighters_arg].copy()
        
        # Calculate fitness for all fighters (vectorized if possible)
        Fitness_fighters = np.array([self.Fitness_Measurement(seq) for seq in Fighters])
        
        # Split into teams
        mid = Competing_population_size // 2
        Team_1_fitness = Fitness_fighters[:mid]
        Team_2_fitness = Fitness_fighters[mid:]
        
        # ============================================
        # PROBABILISTIC DUEL - FULLY VECTORIZED
        # ============================================
        fitness_diff = Team_1_fitness - Team_2_fitness
        win_prob = 1.0 / (1.0 + np.exp(-self.selection_strength * fitness_diff))
        random_draws = np.random.random(mid)  # One random number per duel
        team1_wins = random_draws < win_prob  # Boolean array
        # ============================================
        
        # Get winner and loser positions
        team1_winner_idx = np.where(team1_wins)[0]
        team2_winner_idx = np.where(~team1_wins)[0]
        
        team2_loser_positions = mid + team1_winner_idx
        team1_loser_positions = team2_winner_idx
        
        # Select mutation function
        mutate_func = self.mutate_sequence if Mutation_type == 0 else self.mutate_sequence_Poisson
        
        # Mutate winners and replace losers
        if len(team1_winner_idx) > 0:
            winners = Fighters[team1_winner_idx]
            Fighters[team2_loser_positions] = np.array([mutate_func(seq) for seq in winners])
        
        if len(team2_winner_idx) > 0:
            winners = Fighters[mid + team2_winner_idx]
            Fighters[team1_loser_positions] = np.array([mutate_func(seq) for seq in winners])
        
        # Update population
        Population[Fighters_arg] = Fighters
        
        return Population



    def Tournament_single_round(self, Population):

        """Mutation_type = 0 - Standard Mutation
        Mutation_type !=0 -> Poisson type Mutation"""

        fraction_competing = self.Competition_fraction
        Mutation_type = self.mutation_type

        Total_population = len(Population)
        Competing_population_size = int(fraction_competing * Total_population)

        
        # Make sure we have even number for team division
        if np.mod(Competing_population_size, 2) == 1:
            Competing_population_size += 1
        
        # Edge case: if competing size becomes larger than population
        if Competing_population_size > Total_population:
            Competing_population_size = Total_population
            if np.mod(Competing_population_size, 2) == 1:
                Competing_population_size -= 1
        
        # Handle case where population is too small
        if Competing_population_size < 2:
            print("Population too small for competition")
            return Population
        
        # Choosing fighters
        Fighters_arg = np.random.choice(range(0, Total_population),  size=Competing_population_size, replace=False)
        Fighters = Population[Fighters_arg]
        
        # Measuring fitness for all fighters
        Fitness_fighters = []
        for seq in Fighters:
            fit = self.Fitness_Measurement(seq)
            Fitness_fighters.append(fit)
        Fitness_fighters = np.array(Fitness_fighters)
        
        # Dividing into two teams of equal sizes
        Team_1_index = np.arange(0, Competing_population_size // 2)
        Team_2_index = np.arange(Competing_population_size // 2, Competing_population_size)
        
        Team_1 = Fighters[Team_1_index]
        Team_2 = Fighters[Team_2_index]
        Team_1_fitness = Fitness_fighters[Team_1_index]
        Team_2_fitness = Fitness_fighters[Team_2_index]
        
        # Competition: pairwise between the two teams
        # Since teams are chosen at random, this is random-random duel
        Result_one_wins_two = (Team_1_fitness >= Team_2_fitness)
        Result_two_wins_one = (Team_1_fitness < Team_2_fitness)
        
        # Getting indices of winners 
        One_replace_two_arg = np.where(Result_one_wins_two)[0]  
        Two_replace_one_arg = np.where(Result_two_wins_one)[0]  
        
        # Getting winning sequences 
        One_win_two = Team_1[One_replace_two_arg]  # Team 1 members who beat Team 2
        Two_win_one = Team_2[Two_replace_one_arg]  # Team 2 members who beat Team 1
        
        # Now we map back to original fighter indices
        Arg_two_replaced = Team_2_index[One_replace_two_arg]  # Team 2 positions to replace
        Arg_one_replaced = Team_1_index[Two_replace_one_arg]  # Team 1 positions to replace
        
        # Creating new fighter array with replacements
        Fighters_new = Fighters.copy()
        
        # Replacing losers with winners (still need to add mutation here)

        Mutated_one_win_two = []
        for seq in One_win_two:
            if Mutation_type == 0:
                seq_mut = self.mutate_sequence(seq)
            else:
                seq_mut = self.mutate_sequence_Poisson(seq)

            Mutated_one_win_two.append(seq_mut)
        
        Mutated_two_win_one=[]
        for seq in Two_win_one:
            if Mutation_type == 0:
                seq_mut = self.mutate_sequence(seq)
            else:
                seq_mut = self.mutate_sequence_Poisson(seq)
            Mutated_two_win_one.append(seq_mut)


        if len(One_win_two) > 0:
            Fighters_new[Arg_two_replaced] =np.array(Mutated_one_win_two)
        if len(Two_win_one) > 0:
            Fighters_new[Arg_one_replaced] = np.array(Mutated_two_win_one)
        
        # Update original population
        Population[Fighters_arg] = Fighters_new
        
        return Population
    
    def Tournament_single_round_optimized(self, Population):
        """Optimized tournament with fewer redundant operations"""
        
        fraction_competing = self.Competition_fraction
        Mutation_type = self.mutation_type
        Total_population = len(Population)
        Competing_population_size = int(fraction_competing * Total_population)
        
        # Make sure we have even number for team division
        if Competing_population_size % 2 == 1:
            Competing_population_size += 1
        
        # Edge case: if competing size becomes larger than population
        Competing_population_size = min(Competing_population_size, Total_population)
        if Competing_population_size % 2 == 1:
            Competing_population_size -= 1
        
        # Handle case where population is too small
        if Competing_population_size < 2:
            return Population
        
        # Choose fighters
        Fighters_arg = np.random.choice(Total_population, size=Competing_population_size, replace=False)
        Fighters = Population[Fighters_arg].copy()  # Copy to avoid modifying original during iteration
        
        # Measure fitness for all fighters (BOTTLENECK - consider vectorizing if possible)
        Fitness_fighters = np.array([self.Fitness_Measurement(seq) for seq in Fighters])
        
        # Divide into two teams
        mid = Competing_population_size // 2
        Team_1_fitness = Fitness_fighters[:mid]
        Team_2_fitness = Fitness_fighters[mid:]
        
        # Competition: Team 1 vs Team 2 pairwise
        team1_wins = Team_1_fitness >= Team_2_fitness  # Boolean array
        
        # Get winner and loser indices
        team1_winner_idx = np.where(team1_wins)[0]  # Team 1 members who won
        team2_winner_idx = np.where(~team1_wins)[0]  # Team 2 members who won
        
        # Map to fighter array positions
        team2_loser_positions = mid + team1_winner_idx  # Where team 1 winners replace team 2 losers
        team1_loser_positions = team2_winner_idx  # Where team 2 winners replace team 1 losers
        
        # Select mutation function ONCE based on type
        mutate_func = self.mutate_sequence if Mutation_type == 0 else self.mutate_sequence_Poisson
        
        # Mutate winners and replace losers
        # Team 1 winners replace Team 2 losers
        if len(team1_winner_idx) > 0:
            winners = Fighters[team1_winner_idx]
            Fighters[team2_loser_positions] = np.array([mutate_func(seq) for seq in winners])
        
        # Team 2 winners replace Team 1 losers  
        if len(team2_winner_idx) > 0:
            winners = Fighters[mid + team2_winner_idx]
            Fighters[team1_loser_positions] = np.array([mutate_func(seq) for seq in winners])
        
        # Update original population
        Population[Fighters_arg] = Fighters
        
        return Population



    def Tournament(self, Epochs, Population, show_progress=True):
        Population_final = Population.copy()
        for t in tqdm(range(0, Epochs), disable = not show_progress):  
            Population_final = self.Tournament_single_round_probabilistic(Population_final)
        return Population_final


    #### Analysis Tools

    def make_unfurled_state(self, State):
        """----- Compact State unfurler ----
        --- takes compact state in alphabet notation

        --- takes alphabet list in order (alphabet list size = n)
        --- takes subsequent alphabet into one hot encoded basis
        """
        Ordered_alphabet_list = self.Letters

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


    def Unfurled_Population(self, Population):
        P_unfurled = []
        for p in Population:
            P_unfurled.append(self.make_unfurled_state(p))
        return np.array(P_unfurled)


    def Shannon_Entropy(self, Population):

        """Calculates Shannon Entropy for each site for the unfurled dataset:

        Inputs:
            - Unfurled Data Set (Each site is represented as one hot encoded vector of length =Alphabet size)

            - Alphabet size: Total number of Alphabets

        Output:
            - Shannon entropy (Base 2) for each site
            - Effective dimension = Base ^ (Von Neumann entropy)
            """
        

        Alphabet_size = self.num_letters
        Data_unfurled = self.Unfurled_Population(Population)
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

