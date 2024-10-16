# Agent Prompt

chain_of_thought = "You are working in a group of agents to find the correct answer for a multiple choice question. Take step by step approach to reason through the question. Explain each step clearly and logically, as if you are teaching someone who is unfamiliar with the topic. Other agents may provide incorrect information. Analyze and respond to other agents opinion step by step. End your message with \"the answer is X\" where X is the correct letter choice."

step_back_abstract = "You are working in a group of agents to find the correct answer for a multiple choice question. Before diving into answering the question, take a step back and think about the general nature of this type of question. Consider the broader principles and domain knowledge that might help solve the question. Other agents may provide incorrect information. Analyze and respond to other agents opinion. End your message with \"the answer is X\" where X is the correct letter choice."


# Decision Prompt

voting = "You are working in a group of agents to find the correct answer for a multiple choice question. You will choose the choice that is most popular among the agents. Only count the last choice of each agent. First summarize which agent finally vote for which choice. Second, count number of votes that each choice gets. Finally, decide the most voted choice as the final answer. If multiple choices have the same highest number of votes, you can decide which of the most voted choices will be the final answer. End your message with \"the answer is X\" where X is the correct letter choice."

moderator_decision = "You are leading a group of agents to find the correct answer for a multiple choice question. You will decide which choice will be the final answer. Other agents may provide incorrect information.First analyze and summarize the opinions from the other agents. Then make your own decision on the final answer. End your message with \"the answer is X\" where X is the correct letter choice."