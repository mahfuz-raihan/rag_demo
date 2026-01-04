import json
import os
from main import app # Import the compiled graph

def run_evaluation_suite(test_cases):
    """
    Runs a list of test questions and saves the results.
    """
    results = []
    
    print(f"--- Starting Evaluation on {len(test_cases)} cases ---")
    
    for test in test_cases:
        print(f"Testing: {test['question']}")
        
        inputs = {
            "question": test["question"], 
            "retry_count": 0,
            "documents": [],
            "generation": "",
            "reflection": ""
        }
        
        # We need to track the full state across the stream
        full_state = inputs.copy()
        
        # Execute the graph
        try:
            for output in app.stream(inputs):
                for node_name, state_update in output.items():
                    # Update our local tracking of the state with new values from nodes
                    full_state.update(state_update)
            
            # Extract actual text for debugging
            retrieved_text = [doc.page_content for doc in full_state.get("documents", [])]
            
            results.append({
                "question": test["question"],
                "expected_info": test.get("expected_info", "N/A"),
                "actual_generation": full_state.get("generation"),
                "retries_needed": full_state.get("retry_count", 0),
                "final_reflection": full_state.get("reflection"),
                "context_found": len(retrieved_text) > 0,
                "debug_retrieved_context": retrieved_text # Added for troubleshooting
            })
        except Exception as e:
            print(f"Error processing {test['question']}: {e}")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("--- Evaluation Complete. Results saved to eval_results.json ---")

if __name__ == "__main__":
    # Define your "Gold Standard" questions here
    test_queries = [
        {"question": "Who is the author of the paper?", "expected_info": "Names of authors"},
        {"question": "What is the main methodology used?", "expected_info": "Specific techniques mentioned"},
        {"question": "What are the key results?", "expected_info": "Statistical findings or conclusions"}
    ]
    
    run_evaluation_suite(test_queries)