def modify_list_value(lst):
    lst[1] = -1  # Modifies the second element of the list
    print("Inside function:", lst)

# Original list
my_list = [1, 2, 3]
modify_list_value(my_list)
print("After function call:", my_list)  # Will show [1, -1, 3]