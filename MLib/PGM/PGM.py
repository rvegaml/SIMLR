import numpy as np

'''
------------------------------------------------
PGM general code
------------------------------------------------
'''
class Factor():
    def __init__(self, variables, cardinality, values=[]):
        '''
        This function initializes a factor. A factor is defined by three different arrays:
        - variables: Numpy array that contains the id of the variables in the scope of the factor. The id is a number
            between 0 and # variables.
        - cardinality: Numpy array that contains the cardinality of the variables in the scope of the factor.
            The order of the entries is the same than the order in variables.
        - values: Value of every assignment in the factor. It is a numpy array with prod(cardinality) entries,
            one per every combination of values.
        '''

        self.variables = variables
        self.cardinality = cardinality

        # Compute the number of entries that the factor should have:
        num_entries = np.int32(np.prod(self.cardinality))
        if len(values) > 0:
            if len(values) == num_entries:
                self.values = values
            else:
                # print('Initializing factor with zeros')
                self.values = np.zeros(num_entries)
        else:
            # print('Initializing factor with zeros')
            self.values = np.zeros(num_entries)

        # Create the one-hot encoding object
        # self.onehot_encoder = OneHotEncoder(n_values=np.int16(num_entries), sparse=False)
        # self.onehot_encoder.fit(np.reshape(range(num_entries), (-1,1)))

        # Create the array that converts assignment to index
        temp = np.hstack([self.cardinality, 1])
        temp = np.flip(temp, axis=0)
        temp = np.flip(np.cumprod(temp), axis=0)

        self.convert_a_to_i = temp

        # Create a dictionary that maps every variable to an index
        var_to_ind = dict()
        for i in range(len(variables)):
            var_to_ind[variables[i]] = i

        self.var_to_ind = var_to_ind

    def index_to_assignment(self, index):
        # Extract the vector that contains the cumulative product of the cardinality
        temp = self.convert_a_to_i[1:]

        # Transform the vectors into matrices (This is needed to process several indexes
        # at the same time.)
        temp = np.matmul(np.ones([len(index), 1]), np.reshape(temp,[1,-1]))
        temp_index = np.matmul(np.reshape(index,[-1,1]), np.ones([1, temp.shape[1]]))
        temp_cardinality = np.matmul(np.ones([len(index), 1]), np.reshape(self.cardinality,[1,-1]))

        # Convert the index into the actual assignment
        temp = np.mod(np.floor(np.divide(temp_index,temp)), temp_cardinality)

        return np.int8(temp)

    def assignment_to_index(self, assignment):
        # Function that returns the index (in the values vector) of the given assignment.
        # Assignment is an array with len(self.variables) entries
        temp_card = self.convert_a_to_i[1:]

        index = np.sum(temp_card*assignment, axis=1)

        return np.reshape(index, (-1,1))

    def full_assignment_to_index(self, x):
        # Function that returns the index (in the values vector) of the given assignment.
        # x is an array that contains the entire instance
        assignment = x[:,self.variables]
        return self.assignment_to_index(assignment)


    def get_value_assignment(self, assignment):
        index = self.assignment_to_index(assignment)
        index = np.reshape(index, (-1))

        return np.reshape(self.values[index], (-1,1))

    def get_value_full_assignment(self, x, cardinality):
        continuous_card = (cardinality == 1)
        temp_x = np.array(x)
        temp_x[:, continuous_card] = 0
        assignment = np.int16(temp_x[:,self.variables])

        return self.get_value_assignment(assignment)

    def print_CPT(factor):
        '''
        Function that prints all the assignments ni the factor along with their probabilities
        '''
        assignments = factor.index_to_assignment(list(range(np.prod(factor.cardinality))))
        num_assignments, num_var = assignments.shape
        CPT = np.zeros((num_assignments, num_var+1))
        CPT[:, 0:num_var] = assignments
        CPT[:, -1] = factor.values

        print(CPT)

class Factor_Given_Continuous():
    def __init__(self, variables, cardinality, fn):
        '''
        This function initializes a factor. A factor is defined by three different arrays:
        - variables: Numpy array that contains the id of the variables in the scope of the factor. The id is a number
            between 0 and # variables.
        - cardinality: Numpy array that contains the cardinality of the variables in the scope of the factor.
            The order of the entries is the same than the order in variables.
        - fn: Function to compute the probability of the first variable given the other ones.
        '''

        self.variables = variables
        self.cardinality = cardinality
        self.fn = fn

    def get_value_assignment(self, assignment):
        cpt = self.fn(assignment).numpy()[0]

        return cpt

def FactorProduct(factor_A, factor_B):
    # This function performs the factor product operation. The resulting factor has the entries in ascending order
    # of the variables id.

    if len(factor_A.variables) == 0:
        if len(factor_A.values) == 0:
            factor_C = Factor(factor_B.variables, factor_B.cardinality, factor_B.values)
        else:
            factor_C = Factor(factor_B.variables, factor_B.cardinality, factor_B.values*factor_A.values)

    if len(factor_B.variables) == 0:
        if len(factor_B.values) == 0:
            factor_C = Factor(factor_A.variables, factor_A.cardinality, factor_A.values)
        else:
            factor_C = Factor(factor_A.variables, factor_A.cardinality, factor_B.values*factor_A.values)

    # Set the variables present on the new factor
    var_C = np.union1d(factor_A.variables, factor_B.variables)

    # Identify the indexes of A and B in C
    map_A = np.zeros(len(factor_A.variables), dtype=np.int16)
    counter = 0
    for var in factor_A.variables:
        map_A[counter] = np.where(np.equal(var_C, var))[0][0]
        counter +=1

    map_B = np.zeros(len(factor_B.variables), dtype=np.int16)
    counter = 0
    for var in factor_B.variables:
        map_B[counter] = np.where(np.equal(var_C, var))[0][0]
        counter += 1

    # Set the cardinality of factor C
    card_C = np.zeros(len(var_C), dtype=np.int16)
    card_C[map_A] = factor_A.cardinality
    card_C[map_B] = factor_B.cardinality

    # Create the new factor C
    factor_C = Factor(var_C, card_C)

    # Fill the CPT
    assignments = factor_C.index_to_assignment(list(range(np.prod(factor_C.cardinality))))
    index_A = factor_A.assignment_to_index(assignments[:, map_A])
    index_B = factor_B.assignment_to_index(assignments[:, map_B])

    # To avoid underflow problems, make the multiplication in the log space
    new_prob = np.add(np.log(factor_A.values[index_A]), np.log(factor_B.values[index_B]))
    new_prob = np.exp(new_prob)

    factor_C.values = np.reshape(new_prob, (-1))

    return factor_C

def FactorMarginalization(factor, var_id, operation=np.sum):
    ''' It marginalizes the var_id from the CPT using the operation defined in 'operation'.
    The function returns the unnormalized new factor without the variable to be marginalized.
    '''

    # Find the index of the variable to marginalize
    c_index = factor.var_to_ind[var_id]

    # Find the number of possible values that this variable might take
    c_card = factor.cardinality[c_index]

    # Create a new factor without the variable to marginalize
    new_variables = np.hstack([factor.variables[0:c_index], factor.variables[c_index+1:]])
    new_card = np.hstack([factor.cardinality[0:c_index], factor.cardinality[c_index+1:]])
    new_card = np.int16(new_card)
    new_factor = Factor(new_variables, new_card)

    # Find all the possible assignments of the new factor
    num_new_assignments = np.prod(new_card)
    new_possible_assignments = new_factor.index_to_assignment(list(range(num_new_assignments)))

    # Find all the possible assignments without the variable to marginalize
    num_assignments = np.prod(factor.cardinality)
    possible_assignments = factor.index_to_assignment(list(range(num_assignments)))
    possible_assignments = np.hstack([possible_assignments[:,0:c_index], possible_assignments[:,c_index+1:]])

    # Fill the new CPT
    for assignment in new_possible_assignments:
        i = new_factor.assignment_to_index(np.array([assignment]))
        prob_array = np.zeros(c_card)
        counter = 0
        for j in range(possible_assignments.shape[0]):
            line = possible_assignments[j]
            if np.array_equal(assignment, line):
                prob_array[counter] = factor.values[j]
                counter = counter + 1

        val = operation(prob_array)
        new_factor.values[i] = val

    return new_factor

def multiplyFactors(FactorList):
    num_factors = len(FactorList)
    if num_factors == 1:
        return FactorList[0]
    elif num_factors > 1:
        temp = FactorProduct(FactorList[0], FactorList[1])
        for i in range(2, num_factors):
            temp = FactorProduct(temp, FactorList[i])
        
        return temp
    
def marginalizeVariables(factor, var_list):
    new_Factor = Factor(factor.variables, factor.cardinality, factor.values)
    
    for element in var_list:
        new_Factor = FactorMarginalization(new_Factor, element)
    
    return new_Factor

'''
-------------------------------------------------
PGM for SIMLR
-------------------------------------------------
'''
def compute_uncertainty_given_policies(policy_1, policy_2, policy_3):
    
    variables = np.array([1])
    cardinality = np.array([3])
    Factor_A = Factor(variables, cardinality, policy_1)

    variables = np.array([2])
    cardinality = np.array([3])
    Factor_B = Factor(variables, cardinality, policy_2)

    variables = np.array([3])
    cardinality = np.array([3])
    Factor_C = Factor(variables, cardinality, policy_3)

    variables = np.array([4, 1, 2, 3])
    cardinality = np.array([3, 3, 3, 3])
    values = np.array([1,1,1, 1,1,1, 1,1,0, 1,1,1, 1,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0,
                       0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,
                       0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,1, 1,1,1, 0,1,1, 1,1,1, 1,1,1])
    Factor_D = Factor(variables, cardinality, values)
    
    Factors = [Factor_A, Factor_B, Factor_C, Factor_D]
    new_Factor = multiplyFactors(Factors)
    
    P_D = marginalizeVariables(new_Factor, [1,2,3])

    return P_D.values

def compute_probability_policy_change(Factors, days, alpha, percentage):
    '''
    days: Days since last change
    alpha: angle of the slope of the last 10 days (Hospitalization)
    percentage: percentage of Hospital beds currently occupied
    '''
    # Get the factors
    openness_to_change = Factors[0]
    need_to_change = Factors[1]
    change_policy = Factors[2]
        
    # Check the possible values of the change_policy
    cardinality = change_policy.cardinality[0]
    possible_values_change_policy = np.arange(cardinality)
    possible_values_need_to_change = np.arange(3)
    possible_values_openness_to_change = np.arange(openness_to_change.cardinality[0])
    
    uncalibrated_prob_policy_change = np.zeros(cardinality)
    
    for x_change_policy in possible_values_change_policy:
        for x_openness_to_change in possible_values_openness_to_change:
            for x_need_to_change in possible_values_need_to_change:
                instance_change_policy = np.array([
                    [x_change_policy, x_openness_to_change, x_need_to_change]
                ])
                P_Change_Policy = change_policy.get_value_assignment(instance_change_policy)[0][0]
                
                instance_opennes_to_change = np.array([
                    [x_openness_to_change, days]
                ])
                P_Openness = openness_to_change.get_value_assignment(instance_opennes_to_change)[0][0]
                
                P_need_to_change = need_to_change.get_value_assignment(
                    np.array([[alpha, percentage]]))[x_need_to_change]
                
                uncalibrated_prob_policy_change[x_change_policy] += (
                    P_Change_Policy*P_Openness*P_need_to_change
                )
    
    Z = np.sum(uncalibrated_prob_policy_change)
    prob_policy_change = uncalibrated_prob_policy_change / Z
    
    return prob_policy_change

def check_policy_change_v3(start_indx, start_index_policy, Params_dict, region):
            
    base_policy = Params_dict[region]['Policy'][start_index_policy-1]
    base_policy_level_C2 = np.argmax(base_policy[0:4])
    base_policy_level_C6 = np.argmax(base_policy[4:8])
    base_policy_level_C4 = np.argmax(base_policy[8:])

    # The default value is that there is no change in policy
    change_policy_flag = np.array([0, 1, 0])
    for j in range(7):
        if start_index_policy+j < start_indx:
            c_policy = Params_dict[region]['Policy'][start_index_policy+j]

            if np.sum(np.abs(c_policy - base_policy)) > 0:
                c_policy_level_C2 = np.argmax(c_policy[0:4])
                c_policy_level_C6 = np.argmax(c_policy[4:8])
                c_policy_level_C4 = np.argmax(c_policy[8:])

                if ((c_policy_level_C2 > base_policy_level_C2) or 
                    (c_policy_level_C4 > base_policy_level_C4) or
                    (c_policy_level_C6 > base_policy_level_C6)):
                    change_policy_flag = np.array([0, 0, 1])
                elif ((c_policy_level_C2 < base_policy_level_C2) or 
                      (c_policy_level_C4 > base_policy_level_C4) or
                    (c_policy_level_C6 < base_policy_level_C6)):
                    change_policy_flag = np.array([1, 0, 0])

                break
        else:
            change_policy_flag = None
    
    return change_policy_flag