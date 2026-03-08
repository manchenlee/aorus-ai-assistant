from config import Config

def validate_query(user_query):
    score = 0
    for char in user_query:
        if ord(char) <= 127:
            score += 1 
        else:
            score += 3 
    
    if score > Config.MAX_CHARS:
        return False, f"[Error]: 輸入過長，請精簡問題後再試一次。 Current input query is too long. Please provide a more concise version and try again. (Current character score: {score}/{Config.MAX_CHARS})"
    return True, user_query