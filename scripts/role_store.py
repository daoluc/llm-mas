role_2 = [
   "Mathematician",
   "Economist"
]

role_4 = [
   "Mathematician",
   "Economist",
   "Engineer",
   "Medical Doctor"
]

role_6 = [
   "Mathematician",
   "Economist",
   "Engineer",
   "Medical doctor",
   "Historian",
   "Philosopher"
]

def get_roles(num_roles):
    """
    Returns a list of roles based on the given number.

    Args:
        num_roles (int): The number of roles to return.

    Returns:
        list: A list of role names.

    Raises:
        ValueError: If the number of roles is not 2, 4, or 6.
    """
    if num_roles == 2:
        return role_2
    elif num_roles == 4:
        return role_4
    elif num_roles == 6:
        return role_6
    else:
        raise ValueError("Number of roles must be 2, 4, or 6.")
