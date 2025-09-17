# Step 4: Rule-Based Engine for Eligibility Checks
def check_eligibility(user_input, entities, schemes):
    matched_schemes = []
    user_input = user_input.lower()

    # Keywords for categories
    keyword_mapping = {
        "farmer": ["agriculture", "farmer", "crop", "kisan"],
        "health": ["health", "hospital", "insurance"],
        "women": ["women", "female", "girl"],
        "education": ["education", "student", "school"],
        "housing": ["housing", "house", "awas"],
        "startup": ["startup", "business", "entrepreneur"]
    }

    matched_category = None
    for category, keywords in keyword_mapping.items():
        for keyword in keywords:
            if keyword in user_input:
                matched_category = category
                break
        if matched_category:
            break

    if not matched_category:
        return ["Sorry, I couldn't understand your request. Try saying 'farmer', 'health', 'women', 'education', 'housing', or 'startup'."]

    for scheme in schemes:
        if "category" in scheme and matched_category in scheme["category"].lower():
            matched_schemes.append(scheme)
        elif "description" in scheme and matched_category in scheme["description"].lower():
            matched_schemes.append(scheme)
        elif "eligibility" in scheme:
            for key, value in scheme["eligibility"].items():
                if isinstance(value, str) and matched_category in value.lower():
                    matched_schemes.append(scheme)
                    break
                elif isinstance(value, list) and any(matched_category in str(item).lower() for item in value):
                    matched_schemes.append(scheme)
                    break

    # Additional eligibility checks using entities
    if entities["occupation"] or entities["location"] or entities["income"]:
        filtered_schemes = []
        for scheme in matched_schemes:
            eligibility = scheme["eligibility"]
            match = True
            if entities["occupation"] and "occupation" in eligibility:
                if entities["occupation"] not in str(eligibility["occupation"]).lower():
                    match = False
            if entities["location"] and "location" in eligibility:
                if entities["location"].lower() not in str(eligibility["location"]).lower():
                    match = False
            if entities["income"] and "income" in eligibility:
                if entities["income"] not in str(eligibility["income"]).lower():
                    match = False
            if match:
                filtered_schemes.append(scheme)
        return filtered_schemes
    return matched_schemes

# Step 5: Simulate Seq2Seq Response (Simplified for Demo)
def simulate_seq2seq_response(user_input, lang):
    # For demo purposes, we simulate a response based on language
    if lang == "ta":
        return "நான் உங்களுக்கு உதவ முடியும். எந்த வகையான ஆதரவு தேவை?"
    elif lang == "hi":
        return "मैं आपकी मदद कर सकता हूँ। आपको किस तरह की सहायता चाहिए?"
    else:
        return "I can help you. What kind of support do you need?"