import json
import os

class BotPersona:
    def __init__(self, bot_id):
        self.bot_id = bot_id
        self.persona_data = self._load_persona_data()
        self.persona = self._get_persona()
        self.username = self._get_username()

    def _load_persona_data(self):
        """Load persona data from JSON file"""
        current_dir = os.path.dirname(__file__)
        persona_file = os.path.join(current_dir, "bot_personas.json")
        
        try:
            with open(persona_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: bot_personas.json not found")
            return {}

    def _get_persona(self):
        """Get the persona description for this bot"""
        if self.bot_id in self.persona_data:
            return self.persona_data[self.bot_id].get("description", "")
        return "A machine learning researcher who is passionate about understanding and sharing the latest developments in AI."

    def _get_username(self):
        """Get the username for this bot"""
        if self.bot_id in self.persona_data:
            return self.persona_data[self.bot_id].get("username", "")
        return ""

    def format_tweet(self, title, summary, author_text, paper_id):
        """Format a tweet with the bot's persona"""
        tweet = f"📑 {title}" + "\n"
        
        # Add persona perspective if we have a description
        if self.persona:
            persona_intro = self.persona.split(',')[0]  # Get first part of persona description
            tweet += f"From my perspective as {persona_intro}, this is fascinating: {summary}" + "\n"
        else:
            tweet += f"Key finding: {summary}" + "\n"
            
        tweet += f"By {author_text} #DeepLearning #AI" + "\n"
        
        # Ensure tweet is within Twitter's character limit (280)
        if len(tweet) > 280:
            excess = len(tweet) - 280 + 3  # +3 for the ellipsis
            tweet = f"📑 {title}" + "\n"
            if self.persona:
                tweet += f"From my perspective as {persona_intro}, this is fascinating: {summary[:-excess]}..." + "\n"
            else:
                tweet += f"Key finding: {summary[:-excess]}..." + "\n"
            tweet += f"By {author_text} #DeepLearning #AI" + "\n"
            tweet += f"https://arxiv.org/abs/{paper_id}" + "\n"
        
        return tweet 