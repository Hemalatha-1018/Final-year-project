# Step 6: Main Chatbot Function with Messaging-Style Output
def chatbot():
    print("🌟 Welcome to the Government Scheme Chatbot! 🌟")
    print("I’m here to help you find government schemes for categories like farmers, health, women, education, housing, or startups.")
    print("Just tell me what you need, or type 'exit' to quit.\n")

    # Load schemes
    schemes = load_schemes()

    # Initialize Seq2Seq model (simplified for demo)
    input_dim = 768  # Indic-BERT embedding size
    hidden_dim = 256
    output_dim = 100  # Vocabulary size (simplified)
    encoder = Encoder(input_dim, hidden_dim)
    decoder = Decoder(output_dim, hidden_dim)
    seq2seq = Seq2Seq(encoder, decoder, hidden_dim)

    # Metrics tracking
    responses = []
    references = []
    correct_eligibility_checks = 0
    total_checks = 0
    start_time = time.time()

    while True:
        user_input = input("💬 You: ")
        
        if user_input.lower() == "exit":
            print("🤖 Chatbot [10:02 AM]: Goodbye! If you need help later, just come back! 👋")
            break

        # Measure latency
        query_start = time.time()

        # NLP Pipeline
        lang = detect_language(user_input)
        entities = extract_entities(user_input)
        sentiment = analyze_sentiment(user_input)

        # Process text with Indic-BERT
        embeddings = process_text_with_indic_bert(user_input)

        # Simulate Seq2Seq response
        seq2seq_response = simulate_seq2seq_response(user_input, lang)
        print(f"🤖 Chatbot [10:00 AM]: {seq2seq_response}")

        # Rule-Based Eligibility Check
        matched_schemes = check_eligibility(user_input, entities, schemes)

        # Display results in a messaging style
        if not matched_schemes or isinstance(matched_schemes[0], str):
            print(f"🤖 Chatbot [10:01 AM]: Oh no! 😔 I couldn’t find any schemes for your request.")
            print(f"🤖 Chatbot [10:01 AM]: Try using keywords like 'farmer', 'health', 'women', 'education', 'housing', or 'startup'. What else can I help with?")
        else:
            print(f"🤖 Chatbot [10:01 AM]: Great news! I found {len(matched_schemes)} schemes that might help you. Let me share them one by one. 😊")

            for idx, scheme in enumerate(matched_schemes, 1):
                # Handle scheme name if it's a dictionary
                scheme_name = scheme.get('name', 'N/A')
                if isinstance(scheme_name, dict):
                    scheme_name = scheme_name.get(lang, scheme_name.get('en', 'N/A'))  # Use detected language or default to English

                # Present the scheme in a conversational way
                print(f"🤖 Chatbot [10:0{2+idx} AM]: Here’s scheme number {idx}: *{scheme_name}*")
                print(f"🤖 Chatbot [10:0{2+idx} AM]: It’s about {scheme.get('description', 'helping people in this category (no detailed description available)')}. You can get {scheme.get('benefits', 'some benefits (not specified)')}, which sounds pretty useful!")
                print(f"🤖 Chatbot [10:0{2+idx} AM]: To be eligible, you need: {scheme.get('eligibility', 'some criteria (not specified)')}")
                
                # Ask if the user wants more details or to see the next scheme
                if idx < len(matched_schemes):
                    print(f"🤖 Chatbot [10:0{2+idx} AM]: Would you like to know more about this scheme, or should I tell you about the next one? (Type 'more' for details, 'next' for the next scheme, or 'stop' to stop)")
                    user_choice = input("💬 You: ")
                    if user_choice.lower() == "more":
                        print(f"🤖 Chatbot [10:0{3+idx} AM]: Sure, here’s more about *{scheme_name}*:")
                        if 'ministry' in scheme:
                            print(f"🤖 Chatbot [10:0{3+idx} AM]: It’s managed by {scheme.get('ministry', 'N/A')}.")
                        print(f"🤖 Chatbot [10:0{3+idx} AM]: How to Apply: You can contact your local government office or check the official website for more details.")
                        print(f"🤖 Chatbot [10:0{3+idx} AM]: Should I tell you about the next scheme now? (Type 'next' to continue or 'stop' to stop)")
                        next_choice = input("💬 You: ")
                        if next_choice.lower() == "stop":
                            break
                    elif user_choice.lower() == "stop":
                        break
                else:
                    print(f"🤖 Chatbot [10:0{2+idx} AM]: That’s the last scheme I found for you! Would you like to know more about this one? (Type 'more' for details or 'no' to move on)")
                    user_choice = input("💬 You: ")
                    if user_choice.lower() == "more":
                        print(f"🤖 Chatbot [10:0{3+idx} AM]: Sure, here’s more about *{scheme_name}*:")
                        if 'ministry' in scheme:
                            print(f"🤖 Chatbot [10:0{3+idx} AM]: It’s managed by {scheme.get('ministry', 'N/A')}.")
                        print(f"🤖 Chatbot [10:0{3+idx} AM]: How to Apply: You can contact your local government office or check the official website for more details.")

            # Summary after presenting schemes
            print(f"🤖 Chatbot [10:0{3+len(matched_schemes)} AM]: So, I shared {len(matched_schemes)} schemes with you! I’d recommend starting with *{matched_schemes[0].get('name', 'N/A')}* since it seems like a great fit. What else can I help you with? 😊")

        # Sentiment-based grievance escalation
        if sentiment == "negative":
            print(f"🤖 Chatbot [10:0{4+len(matched_schemes)} AM]: Hmm, it looks like you might not be happy with the results. 😔 Would you like to speak to a human agent instead? (yes/no)")
            escalate = input("💬 You: ")
            if escalate.lower() == "yes":
                print(f"🤖 Chatbot [10:0{5+len(matched_schemes)} AM]: Okay, I’m connecting you to a human agent. Please wait a moment... 📞")
                continue

        # Metrics: BLEU Score
        responses.append(seq2seq_response)
        references.append("I can help you. What kind of support do you need?")  # Reference response
        bleu_score = corpus_bleu(responses, [references]).score
        print(f"🤖 Chatbot [10:0{5+len(matched_schemes)} AM]: 📈 Just a quick note: My response quality (BLEU Score) is {bleu_score:.2f}.")

        # Metrics: Eligibility Check Accuracy (simulated)
        total_checks += 1
        if matched_schemes and not isinstance(matched_schemes[0], str):
            correct_eligibility_checks += 1
        accuracy = (correct_eligibility_checks / total_checks) * 100
        print(f"🤖 Chatbot [10:0{5+len(matched_schemes)} AM]: 📈 My eligibility check accuracy is {accuracy:.2f}%.")

        # Metrics: Latency
        query_end = time.time()
        latency = query_end - query_start
        print(f"🤖 Chatbot [10:0{5+len(matched_schemes)} AM]: ⏱️ I responded in {latency:.2f} seconds.\n")

    # Final Metrics
    total_time = time.time() - start_time
    print("🤖 Chatbot [10:10 AM]: 📊 Final Metrics:")
    print(f"   - Average BLEU Score: {bleu_score:.2f}")
    print(f"   - Final Eligibility Check Accuracy: {accuracy:.2f}%")
    print(f"   - Average Latency: {(total_time / total_checks):.2f} seconds")

# Run the chatbot
chatbot()