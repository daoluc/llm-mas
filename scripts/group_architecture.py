from enum import Enum

class Topology(Enum):
    SINGLE     = "one"
    GROUP_CHAT = "gc"
    ONE_ON_ONE = "1on1"
    REFLECTION = "rfl"

class PromptType(Enum):
    CHAIN_OF_THOUGHT = "cot"
    STEP_BACK_ABSTRACTION = "sba"
    MIXED = "mix"

class GroupArchitecture:
    def __init__(
        self,
        topology: Topology,
        group_size: int,
        prompt_type: PromptType,
        assign_role: bool = True
    ):
        self.topology = topology
        self.group_size = group_size
        self.prompt_type = prompt_type        
        self.assign_role = assign_role

    def __str__(self):
        return f"A({self.topology.value}_{self.group_size}_{self.prompt_type.value}_{self.assign_role})"

    def __repr__(self):
        return self.__str__()
