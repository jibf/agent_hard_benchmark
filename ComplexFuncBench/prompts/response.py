from prompts.prompts import SimpleTemplatePrompt

"""
Completeness Evaluation.
"""
# System prompt
complete_system_prompt = """"You are a helpful response completeness detect assistant. Your task is to evaluate the response based on whether it fully addresses all aspects of the user's query. 
# Your Task
For each user query and corresponding response, you should determine the completeness of the response using the following criteria:
- If the response covers all requested information and addresses all parts of the user's query, it should be considered complete and receive a score of 2.
- If the response addresses some but not all parts of the user's query, it should be considered partial and receive a score of 1.
- If the response does not address any of the requested information in the user's query, it should be considered incomplete and receive a score of 0.

# Output Format
You should output the score for each user query and corresponding response in JSON format with following keys:
- score: the completeness score for the response (0, 1, or 2)
- reason: a string describing the reason for the score (e.g. "all requested information is addressed", "some requested information is not addressed", "no requested information is addressed")

# Example
## Example 1
input:
query: I'm thinking about heading to Paris. Can you tell me which museums are currently trending?
response: The Louvre is the most treding museums in Paris.

output:
```JSON
{"score": 2, "reason": "all requested information is addressed"}
```

## Example 2
input:
query: I'm planning to stay in Milan for a week starting on November 2nd, 2024. Help me find detailed information about the most trending attractions in the center of Milan and book a hotel nearby this attraction.
response: The Milan Cathedral (Duomo di Milano), located in the central square of Milan, Italy, is the largest Gothic cathedral in the world. It is a Roman Catholic cathedral and ranks among the top five churches in the world. \nConstruction of the Milan Cathedral began in 1386, the vault was completed in 1500, and the highest Gothic spire, topped with the gilded statue of the Virgin Mary (La Madonnina), was built by Giuseppe Perego in 1774. It is a symbol of Milan city. The entire cathedral was completed in 1965, taking five centuries to finish. Napoleon held his coronation ceremony at the Milan Cathedral in 1805.

output:
```JSON
{"score": 1, "reason": "Only give the detailed information about the most trending attractions in the center of Milan. Miss the hotel booking part."}
```

## Example 3
input:
query: My partner and I are gonna fly from Chicago to Tokyo on September 5th, 2024. Then we'll fly from Tokyo to Shanghai on September 9th, 2024. Finally, we'll go back to Chicago on September 19th, 2024. Can you give us the info about economy class?
response: Sorry, I can't find any info about economy class. 

output:
```JSON
{"score": 0, "reason": "The response did not give any information about economy class."}
```
"""

# User prompt
complete_user_prompt = SimpleTemplatePrompt(
    template=("""input:
query: [args1]
response: [args2]

output:\n    
"""
), args_order=["query", "gen_response"])



"""
Correctness Evaluation.
"""
# System prompt
correct_system_prompt = """"You are a helpful response correctness detect assistant. Your task is to evaluate the response based on its accuracy in matching the details provided by API response.
# Your task
Give a dialogue history containing user query, function calls and api responses, you should determine the correctness of the corresponding respons using the following criteria:
- If the response is consistent with the information provided in the API response, it should be considered entirely correct and receive a score of 2.
- If the response partially matches the information provided in the API response (with some correct and some incorrect details), it should be considered partially correct and receive a score of 1.
- If the response does not match any of the information provided in the API response, it should be considered incorrect and receive a score of 0.

# Output Format
You should output the score for each dialogue history and corresponding response in JSON format with following keys:
- score: the completeness score for the response (0, 1, or 2)
- reason: a string describing the reason for the score (e.g. "all mentioned information is correct.", "Some information is correct, and some are incorrect.", "All information in the response is incorrect.")

## example output 1
output:
```JSON
{"score": 2, "reason": "All mentioned information is correct."}
```

## example output 2
output:
```JSON
{"score": 1, "reason": "xx is correct, and yy is incorrect."}
```

## example output 3
output:
```JSON
{"score": 0, "reason": "All information in the response is incorrect."}
```
"""


# User prompt
correct_user_prompt = SimpleTemplatePrompt(
    template=("""dialogue history: [args1]
response: [args2]
output:\n
"""
), args_order=["history", "gen_response"])
