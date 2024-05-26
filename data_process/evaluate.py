import json
import concurrent.futures

def process_line(line):
    data = json.loads(line)
    reference = data['reference'].lower().translate(str.maketrans('', '', '😠😁😔😕🥳🤬!?.,'))
    response = data['response'].lower().translate(str.maketrans('', '', '😠😁😔😕🥳🤬!?.,'))
    return reference in response

def evaluate_accuracy(file_path):
    correct = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        total = len(lines)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_line, lines))

        correct = sum(results)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy
    
file_path = '/home/jye/learn/LLM-Factory/output/emotion_recognition/IEMOCAP-llama3_chat/IEMOCAP_0_8.jsonl'
evaluate_accuracy(file_path)