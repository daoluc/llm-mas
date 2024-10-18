import unittest
from setup_groupchat import create_group_chat
from autogen import UserProxyAgent

class TestGroupChat(unittest.TestCase):
    def test_create_group_chat(self):
        # Test parameters
        num_debaters = 2
        roles = ["Scientist", "Philosopher"]
        prompt = "Engage in a thoughtful debate about the nature of consciousness."
        decision_prompt = "Summarize the key points of the debate and provide a conclusion."
        debate_rounds = 2

        # Create the group chat
        chat_manager = create_group_chat(
            num_debaters,
            roles,
            prompt,
            decision_prompt,
            debate_rounds
        )

        # Create a user proxy for interaction
        user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

        # Initiate the chat with a message
        initial_message = "What is consciousness and how does it arise?"
        last_message = user_proxy.initiate_chat(
            chat_manager,
            message=initial_message,
            silent=True
        )

        # Assert that we received a response
        self.assertIsNotNone(last_message)
        self.assertIsInstance(last_message, str)
        self.assertTrue(len(last_message) > 0)

        # You might want to add more specific assertions based on expected content

if __name__ == '__main__':
    unittest.main()
