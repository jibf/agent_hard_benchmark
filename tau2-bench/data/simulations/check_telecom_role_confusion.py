#!/usr/bin/env python3
"""
Script to check for role confusion patterns in telecom simulation files.
Specifically looks for "your phone", "your device" and similar patterns where
users speak as if they are the agent.
"""

import json
import sys
import re
from typing import List, Dict, Any

def detect_telecom_role_confusion(content: str) -> bool:
    """Detect if user message shows role confusion by using agent-like language"""
    if not content or not isinstance(content, str):
        return False
        
    content_lower = content.lower()
    
    # Patterns that indicate role confusion - user speaking as if they are the agent
    confusion_patterns = [
        r'\byour\s+(phone|device|computer|laptop|system|account|order|reservation|flight|booking|service|plan|bill|account|number)\b',
        # r'\bi\s+(can|will|am going to|need to)\s+(help|assist|check|look up|find|process|handle|transfer|connect)\b',
        # r'\blet\s+me\s+(check|look|find|help|assist|process|handle|transfer|connect)\b',
        # r'\bi\s+(have|can see|can find|can access|can check)\b',
        # r'\bhere\s+(is|are)\s+(your|the)\b',
        # r'\bi\s+(will|can)\s+(transfer|connect|put)\s+you\b',
        # # r'\bplease\s+hold\s+on\b',
        # # r'\bone\s+moment\s+(please|while)\b',
        # # r'\bi\s+(am|will be)\s+(looking|checking|processing|handling)\b',
        # r'\bthank\s+you\s+for\s+(calling|contacting)\b',
        # r'\bhow\s+(may|can)\s+i\s+help\s+you\b',
        # r'\bis\s+there\s+anything\s+else\s+i\s+can\s+help\s+you\s+with\b',
        # r'\bi\s+(understand|see|can confirm)\s+that\b',
        # r'\baccording\s+to\s+(my|our)\s+(system|records|database)\b',
        # r'\bi\s+(can|will)\s+(process|handle|take care of)\s+(this|that|your request)\b',
        # r'\bunfortunately\s+i\s+(cannot|am unable to|cannot help)\b',
        # r'\bi\s+(apologize|am sorry)\s+for\s+(the|any)\s+(inconvenience|delay)\b',
        # r'\bplease\s+(be|rest)\s+assured\b',
        # r'\bi\s+(will|can)\s+(make sure|ensure|guarantee)\b',
        # r'\bfor\s+(your|security)\s+(purposes|reasons)\b',
        # r'\bi\s+(need|require)\s+to\s+(verify|confirm|check)\b',
        # r'\bthis\s+(is|will be)\s+(processed|handled|taken care of)\b',
        # r'\byou\s+(will|should)\s+(receive|get)\s+(a|an)\s+(email|confirmation|notification)\b',
        # r'\bi\s+(have|can)\s+(successfully|completed|processed)\b',
        # r'\bplease\s+(note|be aware)\s+that\b',
        # r'\bas\s+(a|an)\s+(customer service|service)\s+(representative|agent)\b',
        # r'\bi\s+(am|will be)\s+(your|the)\s+(agent|representative|assistant)\b',
        # # r'\bmy\s+(name\s+is|system\s+shows)\b',
        # r'\bi\s+(am|will)\s+(here\s+to|available\s+to)\s+(help|assist)\b',
        # r'\bwhat\s+(can\s+i\s+do\s+for\s+you|would\s+you\s+like\s+me\s+to\s+help\s+with)\b'
    ]
    
    for pattern in confusion_patterns:
        if re.search(pattern, content_lower):
            return True
    
    return False

def analyze_telecom_file(filepath: str) -> List[Dict[str, Any]]:
    """Analyze a single telecom simulation file for role confusion"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        role_confusion_examples = []
        
        # Check if file has simulations
        if 'simulations' not in data or not data['simulations']:
            print(f"No simulations found in {filepath}")
            return role_confusion_examples
        
        # Check if this is a telecom file
        if 'environment_info' in data and 'domain_name' in data['environment_info']:
            domain = data['environment_info']['domain_name']
            if domain != 'telecom':
                print(f"Warning: This file is for domain '{domain}', not telecom")
        
        print(f"Analyzing {len(data['simulations'])} simulations in {filepath}")
        
        for sim_idx, simulation in enumerate(data['simulations']):
            if 'messages' not in simulation:
                continue
                
            for msg_idx, message in enumerate(simulation['messages']):
                if message.get('role') == 'user' and 'content' in message:
                    content = message['content']
                    if detect_telecom_role_confusion(content):
                        role_confusion_examples.append({
                            'file': filepath,
                            'simulation_index': sim_idx,
                            'message_index': msg_idx,
                            'content': content,
                            'task_id': simulation.get('task_id', 'unknown'),
                            'simulation_id': simulation.get('id', 'unknown')
                        })
        
        return role_confusion_examples
    
    except Exception as e:
        print(f'Error processing {filepath}: {e}')
        return []

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_telecom_role_confusion.py <json_file_path>")
        print("Example: python check_telecom_role_confusion.py telecom_simulation.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print(f"Checking for role confusion patterns in: {filepath}")
    print("Looking for patterns like 'your phone', 'your device', 'I can help', etc.")
    print("=" * 80)
    
    examples = analyze_telecom_file(filepath)
    
    print(f"\nFound {len(examples)} role confusion examples")
    
    if examples:
        print("\n=== ROLE CONFUSION EXAMPLES ===")
        for i, example in enumerate(examples):
            print(f"\n--- Example {i+1} ---")
            print(f"Task ID: {example['task_id']}")
            print(f"Simulation ID: {example['simulation_id']}")
            print(f"Message Index: {example['message_index']}")
            print(f"Content: {example['content']}")
            print("-" * 80)
    else:
        print("No role confusion examples found.")

if __name__ == '__main__':
    main()
