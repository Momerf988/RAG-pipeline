# PHASE 2: User prompt input
class UserQueryProcessor:
    def __init__(self, embedder_model):
        self.embedder = embedder_model

    def get_user_preferences(self):
        user_prompt = input("\nWhat are we here to learn today? :)").strip()
        while not user_prompt:
            print("Please enter a question or topic to learn about.")
            user_prompt = input("\nWhat are we here to learn today?\n> ").strip() #why not worked here?
        detail_level = input("\nHow detailed would you like the answer to be (e.g., brief, elaborate, in-depth)?\n").strip()
        valid_detail_levels = ["brief", "elaborate", "in-depth"]
        if detail_level not in valid_detail_levels:     #Q: why replaced if with while? A: to avoid infinite loop if user keeps entering invalid input
            detail_level = 'brief'
        return user_prompt, detail_level
    
    def vectorize_query(self, user_prompt):
        embedded_user_query = self.embedder.encode(user_prompt)
        return embedded_user_query.tolist()