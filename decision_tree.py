import math
from collections import namedtuple

# Decision tree algorith in the textbook
'''
function Decision_Tree_Learning(example, attributes, parents_examples)
    if examples is empty:
        return Plurality_Value(parent_examples)
    else if all examples have the same classification:
        return the classification
    else if attributes is empty:
        return Plurality_Value(examples)
    else:
        A = argmax Importance(a, examples) where a is an element of attributes
        tree = a new decision tree with root test A
        for vk in A:
            exs = {e : e is an element of examples and e.A = vk}
            subtree = Decision_Tree_Learning(exs, attributes - A, examples)
            add a branch to tree with label (A = vk) and subtree called subtree
    return tree
'''

# Functions to create
# 1. Plurality_Value => Selects the most common output value among a set of examples, breaking ties randomly
# by output value = classifier (or result)
# 2. Importance (Info Gain)

# divide and conqure => split until dataset is pure (2nd base case)

# Big Idea
# 1. get examples from a person
# 2. divide examples based on info gain
# 3. recurse

class LeafNode:
    '''Classification'''
    examples = []

    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, position):
        return self.examples[position]
        
    def __repr__(self):
        return "LeafNode: {}".format(self.examples)


class DecisionNode:
    '''where decision was made'''

    subset = []

    def __init__(self):
        self.subset = []

    def add(self, subset_to_add):
        '''adding result'''
        self.subset.append(subset_to_add)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, position):
        return self.subset[position]

    def __repr__(self):
        return "Result => {}".format(self.subset)


hold_result = DecisionNode()
chosen_attributes = []


class Question:
    
    def __init__(self, decision_fork, edges):
        self.decision_fork = decision_fork
        self.edges = edges
    
    def __repr__(self):
        return "{} {}".format(self.decision_fork, self.edges)


def decision_tree_learning(examples, attributes, *parent_examples):
    '''decision tree learning algorithm'''
    if not examples:
        return plurality_value(parent_examples)
    elif are_examples_pure(examples):
        result = LeafNode(examples)
        hold_result.add(result)
        return result
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        selected_attribute = split_on_which_attribute(examples, attributes)
        partitioned_examples = partition(selected_attribute, examples)
        chosen_attributes.append(selected_attribute)
        for subset in partitioned_examples:
            decision_tree_learning(
                subset,
                exclude_previous_attribute(attributes, chosen_attributes),
                partitioned_examples)
    return hold_result


def exclude_previous_attribute(attributes, exclude):
    '''exclude an attribute in order to avoid redundancy'''
    for attribute in attributes:
        if attribute in exclude:
            attributes.remove(attribute)
    # print(attributes)
    return attributes


def plurality_value(examples):
    '''select the most common output value
    among a set of examples, breaking ties randomly'''
    unique_output_values = get_unique_classes(examples)

    output_value_dictionary = {unique_value: 0 for unique_value in unique_output_values}

    for each_unique_output_values in unique_output_values:
        count = 0
        for row in examples:
            if row[-1] == each_unique_output_values:
                count += 1
        output_value_dictionary[each_unique_output_values] = count

    most_common_value = max(output_value_dictionary, key=(lambda key: output_value_dictionary[key]))
    return most_common_value


def get_unique_classes(examples):
    '''getting unique values from a list of example(s)'''
    unique_value_list = []
    for row in examples:
        unique_value_list.append(row[-1])
    return set(unique_value_list)


def get_subsets_of_examples(attribute, examples):
    '''Examples that are divided into subsets based on values'''
    values = []
    for row in examples:
        values.append(row[attribute])
    return set(values)


def are_examples_pure(examples):
    '''base case to see if a set or a subset is pure'''
    count_of_class = []
    for each_example in examples:
        count_of_class.append(each_example[-1])
    if len(set(count_of_class)) == 1:
        return True
    else:
        return False


def entropy(examples):
    '''get entropy'''
    unique_classes = get_unique_classes(examples)
    total = len(examples)
    probability_table = []
    for each_class in unique_classes:
        count_for_probability = 0
        for each_example in examples:  # each row
            if each_example[-1] == each_class:
                count_for_probability += 1
        probability = (count_for_probability/total)
        probability_table.append(probability)
    current_entropy = sum(-p * math.log2(p) for p in probability_table)

    return current_entropy


def information_gain(attribute, examples):
    '''calculate information gain at a node'''
    # importance function in the textbook
    # calculate entrophy from each subset
    # and subtract them from parent's entrophy

    list_of_unique_values = get_subsets_of_examples(attribute, examples)
    # print(list_of_unique_values)

    entrophy_list = []
    value_entropy_pair = namedtuple('Value_Entrophy_Pair', 'value entrophy probability')

    print("**** Attribute: {} ****\n".format(attribute))
    for each_value in list_of_unique_values:
        subset = [example for example in examples if example[attribute] == each_value]
        print("Attribute Value: " + str(each_value))
        print("Entrophy value: " + str(entropy(subset)) + "\n")
        probability = float(len(subset)/len(examples))
        entrophy_list.append(value_entropy_pair(each_value, entropy(subset), probability))

    remainder = 0
    for each_entropy_value in entrophy_list:
        remainder += each_entropy_value.entrophy * each_entropy_value.probability
    
    remainder = math.ceil(remainder*10000)/10000
    print("Remainder for attribute {} is: {}\n".format(attribute, remainder))

    attribute_remainder_pair = namedtuple('attribute_remainder_pair', 'attribute remainder')

    return attribute_remainder_pair(attribute, remainder)


def calculate_info_gain_for_each_attribute(remainders, parent_examples):
    information_gain_list = []
    information_gain = namedtuple('Information_Gain', 'attribute gain')
    for each_remainder in remainders:
        # print(entropy(parent_examples))
        gain = math.ceil((entropy(parent_examples) - each_remainder.remainder) * 10000) / 10000
        information_gain_list.append(information_gain(each_remainder.attribute, gain))
    return information_gain_list


def split_on_which_attribute(examples, attributes):
    '''Find the best attribute to split on'''
    information_gain_table = []
    for attribute in attributes:
        if attribute is not None:
            information_gain_table.append(information_gain(attribute, examples))
        else:
            pass

    table_of_remainders = calculate_info_gain_for_each_attribute(information_gain_table, examples)
    max_gain = max([info.gain for info in table_of_remainders])

    split_on = []
    for each_remainder in table_of_remainders:
        if each_remainder.gain == max_gain:
            split_on.append(each_remainder.attribute)

    print("Spliting on {} gives us the maximum info gain of {}".format(split_on[0], max_gain))
    return split_on[0]


def partition(attribute_to_split_on, examples):
    '''partition examples based on an attribute that gives us
    the best information gain'''
    values_attribute_can_take = get_subsets_of_examples(attribute_to_split_on, examples)

    total = []

    for value in values_attribute_can_take:
        rows_partitioned_based_on_value = []
        for row in examples:
            if row[attribute_to_split_on] == value:
                rows_partitioned_based_on_value.append(row)
        total.append(rows_partitioned_based_on_value)

    return total


def simplify_tree(decision_tree):
    '''make result look better'''
    print("\nThe followings are leaf nodes: \n")
    for branch in decision_tree:
        print(branch)
    print("\n")


# def print_tree(decision_tree):
#     '''print tree'''
#     for split_decision in chosen_attributes:
        
#     # return Non


if __name__ == '__main__':
    print("\nWelcome to Decision Tree Algorithm Test")
    text_file_name = input("\nPlease enter a file name: ")
    print("\n")
    training_data = open(text_file_name, 'r')
    examples = [row.split(',') for row in training_data.read().splitlines()]

    clean_examples = []

    for example in examples:
        if len(example) != 0:
            clean_examples.append(example)
        else:
            pass

    attributes = list(range(len(clean_examples[0]) - 1))

    classified_result = decision_tree_learning(clean_examples, attributes)
    simplify_tree(classified_result)

    print("The following attributes were used to split examples (in order): ")
    print(chosen_attributes)